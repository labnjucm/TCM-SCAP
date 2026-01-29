import { NextRequest, NextResponse } from "next/server";
import { verify } from "@/app/lib/jwt";
import { prisma } from "@/app/lib/prisma";

async function requireAdmin(req: NextRequest) {
  const token = req.cookies.get("chemhub_token")?.value;
  if (!token) return null;
  const payload = verify(token);
  if (!payload) return null;

  const me = await prisma.user.findUnique({
    where: { id: payload.uid },
    select: { id: true, role: true },
  });
  if (!me || me.role !== "admin") return null;
  return me;
}

export async function PATCH(req: NextRequest, { params }: { params: { id: string } }) {
  const admin = await requireAdmin(req);
  if (!admin) return NextResponse.json({ ok: false, error: "Forbidden" }, { status: 403 });

  const uid = Number(params.id);
  const body = await req.json();
  const { canDocking, canMD, canOrca } = body;

  // 防止把管理员自己关掉导致无法管理（可选）
  const target = await prisma.user.findUnique({ where: { id: uid }, select: { role: true } });
  if (target?.role === "admin") {
    return NextResponse.json({ ok: false, error: "不允许修改管理员权限" }, { status: 400 });
  }

  const user = await prisma.user.update({
    where: { id: uid },
    data: {
      canDocking: !!canDocking,
      canMD: !!canMD,
      canOrca: !!canOrca,
    },
    select: { id: true, email: true, canDocking: true, canMD: true, canOrca: true },
  });

  return NextResponse.json({ ok: true, user });
}
