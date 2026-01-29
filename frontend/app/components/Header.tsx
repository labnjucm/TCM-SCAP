"use client";

import { useState, useEffect } from "react";

interface User {
  email: string;
  role?: "user" | "admin";
}


export default function Header() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/auth/me")
      .then(res => res.json())
      .then(data => {
        if (data.ok) {
          setUser(data.user);
        }
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });
  }, []);

  const handleLogout = async () => {
    try {
      await fetch("/api/auth/logout", { method: "POST" });
      window.location.reload();
    } catch (error) {
      console.error("退出登录失败", error);
    }
  };

  return (
    <header className="h-16 flex items-center justify-between px-4 border-b bg-white">
      <div className="flex items-center gap-2">
        <img alt="logo" src="/logo.svg" className="w-7 h-7" />
        <span className="font-semibold text-lg">
          {process.env.NEXT_PUBLIC_APP_TITLE || "ChemHub"}
        </span>
      </div>
      
      <div className="flex items-center gap-4">
        {loading ? (
          <span className="text-sm text-gray-400">加载中...</span>
        ) : user ? (
<div className="flex items-center gap-3">
  <span className="text-sm text-gray-700">{user.email}</span>

  {user.role === "admin" && (
    <a
      href="/admin"
      className="text-sm text-gray-600 hover:text-gray-900 underline"
    >
      管理后台
    </a>
  )}

  <button
    onClick={handleLogout}
    className="text-sm text-gray-600 hover:text-gray-900 underline"
  >
    退出登录
  </button>
</div>

        ) : (
          <span className="text-sm text-gray-500">未登录</span>
        )}
        
        <div className="text-sm text-gray-400 border-l pl-4">
          {process.env.NEXT_PUBLIC_FOOTER_NOTE || ""}
        </div>
      </div>
    </header>
  );
}
