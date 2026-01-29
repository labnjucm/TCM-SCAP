import { NextRequest, NextResponse } from 'next/server'
import { verify } from '@/app/lib/jwt'
import { prisma } from '@/app/lib/prisma'

export async function GET(request: NextRequest) {
  try {
    const token = request.cookies.get('chemhub_token')?.value

    if (!token) {
      return NextResponse.json(
        { ok: false, error: '未登录' },
        { status: 401 }
      )
    }

    const payload = verify(token)
    if (!payload) {
      return NextResponse.json(
        { ok: false, error: 'Token 无效或已过期' },
        { status: 401 }
      )
    }

    // 从数据库获取最新用户信息
    const user = await prisma.user.findUnique({
      where: { id: payload.uid },
      select: {
        id: true,
        email: true,
        role: true,
        canDocking: true,
        canMD: true,
        canOrca: true,
        createdAt: true
      }
    })

    if (!user) {
      return NextResponse.json(
        { ok: false, error: '用户不存在' },
        { status: 404 }
      )
    }

    return NextResponse.json({
      ok: true,
      user
    })
  } catch (error) {
    console.error('获取用户信息错误:', error)
    return NextResponse.json(
      { ok: false, error: '服务器错误' },
      { status: 500 }
    )
  }
}


