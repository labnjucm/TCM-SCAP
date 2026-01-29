"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight, ExternalLink, Globe } from "lucide-react";
import { Catalog, CatalogItem } from "../lib/types";

export default function Sidebar({
  catalog,
  onSelect
}: {
  catalog: Catalog;
  onSelect: (item: CatalogItem) => void;
}) {
  const [open, setOpen] = useState<Record<string, boolean>>({});

  return (
    <aside className="w-80 h-[calc(100vh-64px)] overflow-y-auto border-r bg-white">
      {catalog.sections.map((sec) => {
        const isOpen = !!open[sec.title];
        return (
          <div key={sec.title} className="px-3 py-3">
            <button
              className="w-full flex items-center justify-between text-left font-semibold text-gray-700 hover:text-black"
              onClick={() => setOpen({ ...open, [sec.title]: !isOpen })}
            >
              <span>{sec.title}</span>
              {isOpen ? <ChevronDown size={18}/> : <ChevronRight size={18}/> }
            </button>

            {isOpen && (
              <div className="mt-2 space-y-1">
                {sec.items.map((it) => (
                  <button
                    key={it.key}
                    className="w-full text-left px-2 py-1 rounded hover:bg-gray-100 flex items-center gap-2 text-sm"
                    onClick={() => onSelect(it)}
                  >
                    {it.iframeSrc ? <Globe size={16}/> : <ExternalLink size={16}/>}
                    <span>{it.title}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </aside>
  );
}

