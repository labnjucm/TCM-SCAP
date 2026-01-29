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

export async function GET(req: NextRequest) {
  const admin = await requireAdmin(req);
  if (!admin) return NextResponse.json({ ok: false, error: "Forbidden" }, { status: 403 });

  const users = await prisma.user.findMany({
    orderBy: { createdAt: "desc" },
    select: {
      id: true,
      email: true,
      role: true,
      canDocking: true,
      canMD: true,
      canOrca: true,
      createdAt: true,
    },
  });

  return NextResponse.json({ ok: true, users });
}
