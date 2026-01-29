import { NextRequest, NextResponse } from 'next/server'
import bcrypt from 'bcrypt'
import { prisma } from '@/app/lib/prisma'
import { sign } from '@/app/lib/jwt'

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

    // 查找用户
    const user = await prisma.user.findUnique({
      where: { email }
    })

    if (!user) {
      return NextResponse.json(
        { ok: false, error: '邮箱或密码错误' },
        { status: 401 }
      )
    }

    // 验证密码
    const valid = await bcrypt.compare(password, user.passwordHash)
    if (!valid) {
      return NextResponse.json(
        { ok: false, error: '邮箱或密码错误' },
        { status: 401 }
      )
    }

    // 生成 JWT
    const token = sign({
      uid: user.id,
      email: user.email
    })

    // 设置 Cookie
    const response = NextResponse.json({
      ok: true,
      user: {
        id: user.id,
        email: user.email
      }
    })

    const isProduction = process.env.NODE_ENV === 'production'
    response.cookies.set('chemhub_token', token, {
      httpOnly: true,
      secure: isProduction,
      sameSite: 'lax',
      path: '/',
      maxAge: 7 * 24 * 60 * 60 // 7 days
    })

    return response
  } catch (error) {
    console.error('登录错误:', error)
    return NextResponse.json(
      { ok: false, error: '服务器错误' },
      { status: 500 }
    )
  }
}


