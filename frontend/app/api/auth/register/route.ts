import { NextRequest, NextResponse } from 'next/server'
import bcrypt from 'bcrypt'
import { prisma } from '@/app/lib/prisma'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { email, password } = body

    if (!email || !password) {
      return NextResponse.json(
        { ok: false, error: '邮箱和密码不能为空' },
        { status: 400 }
      )
    }

    if (password.length < 8) {
      return NextResponse.json(
        { ok: false, error: '密码至少需要 8 位' },
        { status: 400 }
      )
    }

    // 检查邮箱是否已存在
    const existing = await prisma.user.findUnique({
      where: { email }
    })

    if (existing) {
      return NextResponse.json(
        { ok: false, error: '该邮箱已注册' },
        { status: 400 }
      )
    }

    // 哈希密码
    const passwordHash = await bcrypt.hash(password, 10)

    // 创建用户
    const user = await prisma.user.create({
      data: {
        email,
        passwordHash,
        role: "user",
        canDocking: false,
        canMD: false,
        canOrca: false
      },
      select: { id: true, email: true }
    });

    return NextResponse.json({
      ok: true,
      user: {
        id: user.id,
        email: user.email
      }
    })
  } catch (error) {
    console.error('注册错误:', error)
    return NextResponse.json(
      { ok: false, error: '服务器错误' },
      { status: 500 }
    )
  }
}


