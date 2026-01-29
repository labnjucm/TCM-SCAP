"use client";

import { useState } from "react";
import { Settings } from "lucide-react";
import AuthModal from "./AuthModal";

export default function SettingsFab() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 w-14 h-14 bg-brand text-white rounded-full shadow-lg hover:bg-sky-600 transition-all hover:scale-110 flex items-center justify-center z-40"
        title="设置"
      >
        <Settings size={24} />
      </button>

      <AuthModal isOpen={isOpen} onClose={() => setIsOpen(false)} />
    </>
  );
}


