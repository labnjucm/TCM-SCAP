import jwt from 'jsonwebtoken'

const JWT_SECRET = process.env.JWT_SECRET || 'change_me_in_production'

export interface TokenPayload {
  uid: number
  email: string
}

export function sign(payload: TokenPayload, expDays: number = 7): string {
  return jwt.sign(payload, JWT_SECRET, {
    expiresIn: `${expDays}d`,
    algorithm: 'HS256'
  })
}

export function verify(token: string): TokenPayload | null {
  try {
    const decoded = jwt.verify(token, JWT_SECRET, { algorithms: ['HS256'] })
    return decoded as TokenPayload
  } catch (error) {
    return null
  }
}


