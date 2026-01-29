"use client";

export default function IframePane({ src }: { src: string }) {
  return (
    <div className="w-full h-[calc(100vh-96px)] border rounded-lg bg-white">
      <iframe
        className="w-full h-full"
        src={src}
        sandbox="allow-same-origin allow-scripts allow-forms allow-modals allow-downloads allow-popups allow-popups-to-escape-sandbox"
      />
    </div>
  );
}

