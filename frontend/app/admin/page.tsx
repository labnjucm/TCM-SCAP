"use client";

import { useEffect, useState } from "react";

type UserRow = {
  id: number;
  email: string;
  role: "user" | "admin";
  canDocking: boolean;
  canMD: boolean;
  canOrca: boolean;
};

export default function AdminPage() {
  const [users, setUsers] = useState<UserRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    setError(null);
    const res = await fetch("/api/admin/users");
    const data = await res.json();
    if (!data.ok) {
      setError(data.error || "无权限或未登录");
      setUsers([]);
    } else {
      setUsers(data.users);
    }
    setLoading(false);
  };

  useEffect(() => {
    load();
  }, []);

  const update = async (u: UserRow, patch: Partial<UserRow>) => {
    const res = await fetch(`/api/admin/users/${u.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        canDocking: patch.canDocking ?? u.canDocking,
        canMD: patch.canMD ?? u.canMD,
        canOrca: patch.canOrca ?? u.canOrca,
      }),
    });
    const data = await res.json();
    if (!data.ok) {
      alert(data.error || "更新失败");
      return;
    }
    setUsers((prev) =>
      prev.map((x) => (x.id === u.id ? { ...x, ...data.user } : x))
    );
  };

  if (loading) return <div className="p-6">加载中...</div>;
  if (error) return <div className="p-6 text-red-600">{error}</div>;

  return (
    <div className="p-6 space-y-4">
      <h1 className="text-xl font-semibold">权限管理</h1>

      <div className="overflow-x-auto border rounded">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="text-left p-3">邮箱</th>
              <th className="text-left p-3">角色</th>
              <th className="text-left p-3">分子对接</th>
              <th className="text-left p-3">分子动力学</th>
              <th className="text-left p-3">ORCA</th>
            </tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.id} className="border-t">
                <td className="p-3">{u.email}</td>
                <td className="p-3">{u.role}</td>

                <td className="p-3">
                  <input
                    type="checkbox"
                    disabled={u.role === "admin"}
                    checked={u.canDocking}
                    onChange={(e) => update(u, { canDocking: e.target.checked })}
                  />
                </td>

                <td className="p-3">
                  <input
                    type="checkbox"
                    disabled={u.role === "admin"}
                    checked={u.canMD}
                    onChange={(e) => update(u, { canMD: e.target.checked })}
                  />
                </td>

                <td className="p-3">
                  <input
                    type="checkbox"
                    disabled={u.role === "admin"}
                    checked={u.canOrca}
                    onChange={(e) => update(u, { canOrca: e.target.checked })}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <button
        className="px-3 py-2 rounded bg-gray-900 text-white"
        onClick={load}
      >
        刷新
      </button>
    </div>
  );
}
