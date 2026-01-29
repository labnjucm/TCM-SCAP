"use client";

import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import { CatalogItem } from "./lib/types";
import ReactMarkdown from "react-markdown";
import { useEffect, useMemo, useState } from "react";
import { catalog } from "@/app/config/catalog";

type MeUser = {
  id: number;
  email: string;
  role: "user" | "admin";
  canDocking: boolean;
  canMD: boolean;
  canOrca: boolean;
};

export default function Page() {
  const [me, setMe] = useState<MeUser | null>(null);

  const [selected, setSelected] = useState<CatalogItem | null>(null);
  const [homepageContent, setHomepageContent] = useState<string>("");
  const [detailsContent, setDetailsContent] = useState<string>("");
  const [loading, setLoading] = useState(false);

  // 读取当前登录用户（含权限）
  useEffect(() => {
    fetch("/api/auth/me")
      .then((r) => r.json())
      .then((d) => {
        if (d.ok) setMe(d.user);
        else setMe(null);
      })
      .catch(() => setMe(null));
  }, []);

  // 加载主页介绍内容
  useEffect(() => {
    fetch("/api/content/homepage-intro")
      .then((res) => (res.ok ? res.text() : ""))
      .then((text) => setHomepageContent(text))
      .catch(() => setHomepageContent(""));
  }, []);

  // 当选择工具且有详细说明时，加载详细内容
  useEffect(() => {
    if (selected?.detailsSlug) {
      setLoading(true);
      fetch(`/api/content/${selected.detailsSlug}`)
        .then((res) => (res.ok ? res.text() : ""))
        .then((text) => {
          setDetailsContent(text);
          setLoading(false);
        })
        .catch(() => {
          setDetailsContent("");
          setLoading(false);
        });
    } else {
      setDetailsContent("");
    }
  }, [selected?.detailsSlug]);

  const hasPerm = (p: "docking" | "md" | "orca") => {
    if (!me) return false;
    if (me.role === "admin") return true;
    if (p === "docking") return me.canDocking;
    if (p === "md") return me.canMD;
    return me.canOrca;
  };

  const filteredCatalog = useMemo(() => {
    return {
      sections: catalog.sections
        .map((sec) => ({
          ...sec,
          items: sec.items.filter((it: any) => {
            if (!it.requires) return true;
            return hasPerm(it.requires);
          }),
        }))
        .filter((sec) => sec.items.length > 0),
    };
  }, [me]);

  return (
    <div className="min-h-screen">
      <Header />
      <div className="flex">
        <div className="hidden md:block">
          <Sidebar catalog={filteredCatalog} onSelect={(it) => setSelected(it)} />
        </div>

        <main className="flex-1 p-4">
          {!selected ? (
            <div className="prose prose-slate max-w-none">
              <ReactMarkdown>{homepageContent}</ReactMarkdown>
            </div>
          ) : selected.iframeSrc ? (
            <div className="space-y-4">
              <h2 className="text-xl font-semibold">{selected.title}</h2>

              {selected.intro && (
                <div
                  className="prose max-w-none"
                  dangerouslySetInnerHTML={{
                    __html: selected.intro.replace(/\n/g, "<br/>"),
                  }}
                />
              )}

              {selected.link && (
                <a
                  className="inline-flex items-center gap-2 px-3 py-2 rounded bg-brand text-white hover:bg-sky-600"
                  href={selected.link}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  打开官方链接
                </a>
              )}

              <iframe
                className="w-full h-[calc(100vh-200px)] border rounded"
                src={selected.iframeSrc}
                sandbox="allow-same-origin allow-scripts allow-forms allow-modals allow-downloads allow-popups allow-popups-to-escape-sandbox"
              />
            </div>
          ) : (
            <div className="space-y-4">
              <h2 className="text-xl font-semibold">{selected.title}</h2>

              {selected.intro && (
                <div className="bg-blue-50 border-l-4 border-brand p-4 rounded">
                  <div
                    className="prose max-w-none text-sm"
                    dangerouslySetInnerHTML={{
                      __html: selected.intro.replace(/\n/g, "<br/>"),
                    }}
                  />
                </div>
              )}

              {selected.link && (
                <a
                  className="inline-flex items-center gap-2 px-3 py-2 rounded bg-brand text-white hover:bg-sky-600 transition-colors"
                  href={selected.link}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  打开官方链接
                </a>
              )}

              {selected.detailsSlug && (
                <div className="border-t pt-4">
                  <h3 className="text-lg font-semibold mb-3 text-gray-800">
                    📖 详细说明
                  </h3>

                  {loading ? (
                    <div className="flex items-center justify-center py-12">
                      <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-brand"></div>
                    </div>
                  ) : (
                    <div className="prose prose-slate max-w-none">
                      <ReactMarkdown>{detailsContent}</ReactMarkdown>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
