import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

export async function GET(
  request: NextRequest,
  { params }: { params: { slug: string } }
) {
  try {
    const { slug } = params
    
    // 安全检查：防止路径遍历
    if (slug.includes('..') || slug.includes('/')) {
      return new Response('Invalid slug', { status: 400 })
    }

    const filePath = path.join(process.cwd(), 'app', 'content-md', `${slug}.md`)
    
    if (!fs.existsSync(filePath)) {
      return new Response('Content not found', { status: 404 })
    }

    const content = fs.readFileSync(filePath, 'utf-8')
    
    return new Response(content, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Cache-Control': 'public, max-age=3600'
      }
    })
  } catch (error) {
    console.error('读取内容错误:', error)
    return new Response('Server error', { status: 500 })
  }
}


