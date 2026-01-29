import "./globals.css";
import { ReactNode } from "react";
import SettingsFab from "./components/SettingsFab";

export const metadata = {
  title: process.env.NEXT_PUBLIC_APP_TITLE || "ChemHub",
  description: "Unified portal for docking/MD/ADMET/quantum tools"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>
        {children}
        <SettingsFab />
      </body>
    </html>
  );
}

