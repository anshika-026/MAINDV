export default function AuthLayout({ children }) {
  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-[#f5f6fa] px-4">
      <div className="w-full max-w-sm">
        <div className="flex items-center gap-2 justify-center mb-8">
          <div className="w-8 h-8 rounded-md bg-brand-500 flex items-center justify-center text-white font-bold text-sm">
            D
          </div>
          <span className="font-semibold text-lg text-ink-900">Deco Vision</span>
        </div>
        <div className="card p-7">{children}</div>
      </div>
    </div>
  );
}
