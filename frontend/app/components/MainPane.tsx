"use client";

import { useState } from "react";
import Markdown from "./Markdown";
import IframePane from "./IframePane";
import { Catalog, CatalogItem } from "../lib/types";

function IntroCard({ item }: { item: CatalogItem }) {
  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">{item.title}</h2>
      {item.intro && <Markdown content={item.intro} />}
      {item.link && (
        <a 
          className="inline-flex items-center gap-2 px-3 py-2 rounded bg-brand text-white hover:bg-sky-600" 
          href={item.link} 
          target="_blank"
          rel="noopener noreferrer"
        >
          打开官方链接
        </a>
      )}
      {item.iframeSrc && (
        <div className="pt-2">
          <IframePane src={item.iframeSrc} />
        </div>
      )}
    </div>
  );
}

export default function MainPane({ catalog }: { catalog: Catalog }) {
  const [selected, setSelected] = useState<CatalogItem | null>(null);

  return (
    <div className="flex">
      <div className="flex-1 p-4">
        {!selected ? (
          <div className="text-gray-600">
            请选择左侧分类与工具；支持在同域子路径内嵌你现有的 Gradio 界面和 ADMET 网站。
          </div>
        ) : (
          <IntroCard item={selected} />
        )}
      </div>
    </div>
  );
}

