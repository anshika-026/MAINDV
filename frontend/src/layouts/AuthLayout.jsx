import BrandLogo from "../components/BrandLogo";

export default function AuthLayout({ children }) {
  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-[#f5f6fa] px-4">
      <div className="w-full max-w-sm">
        <div className="flex items-center gap-2 justify-center mb-8">
          <BrandLogo size={36} />
          <span className="font-semibold text-lg text-ink-900">Deco Vision</span>
        </div>
        <div className="card p-7">{children}</div>
      </div>
    </div>
  );
}
