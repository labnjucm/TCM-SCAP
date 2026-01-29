export type PermissionKey = "docking" | "md" | "orca";
export type CatalogItem = {
  key: string;
  title: string;
  intro?: string;
  link?: string;
  iframeSrc?: string; // 若设置则在中间区域内嵌
  detailsSlug?: string; // 对应 content-md/<slug>.md
  requires?: PermissionKey;
};

export type CatalogSection = {
  title: string;
  items: CatalogItem[];
};

export type Catalog = {
  sections: CatalogSection[];
};

