import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import AuthLayout from "../layouts/AuthLayout";
import PasswordField from "../components/PasswordField";
import { useAuth } from "../context/AuthContext";

const STEPS = ["details", "account", "otp"];

export default function Signup() {
  const { signup } = useAuth();
  const navigate = useNavigate();
  const [step, setStep] = useState(0);
  const [form, setForm] = useState({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    otp: "",
  });
  const [loading, setLoading] = useState(false);

  function update(field, value) {
    setForm((f) => ({ ...f, [field]: value }));
  }

  function next(e) {
    e.preventDefault();
    setStep((s) => Math.min(s + 1, STEPS.length - 1));
  }

  async function finish(e) {
    e.preventDefault();
    setLoading(true);
    try {
      await signup({ name: `${form.firstName} ${form.lastName}`.trim(), email: form.email });
      navigate("/dashboard");
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthLayout>
      {step === 0 && (
        <>
          <h1 className="text-lg font-semibold text-ink-900">Get started</h1>
          <p className="text-sm text-slate-500 mt-1 mb-6">Get yourself in within 1 minute</p>
          <form onSubmit={next} className="space-y-4">
            <div>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">First name</label>
              <input
                required
                value={form.firstName}
                onChange={(e) => update("firstName", e.target.value)}
                placeholder="First name"
                className="input-field"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">Last name</label>
              <input
                required
                value={form.lastName}
                onChange={(e) => update("lastName", e.target.value)}
                placeholder="Last name"
                className="input-field"
              />
            </div>
            <button type="submit" className="btn-primary w-full">
              Next
            </button>
          </form>
        </>
      )}

      {step === 1 && (
        <>
          <h1 className="text-lg font-semibold text-ink-900">Create your Account</h1>
          <p className="text-sm text-slate-500 mt-1 mb-6">Set yourself up within 1 minute</p>
          <form onSubmit={next} className="space-y-4">
            <div>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">Work email</label>
              <input
                type="email"
                required
                value={form.email}
                onChange={(e) => update("email", e.target.value)}
                placeholder="Email address"
                className="input-field"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">Create password</label>
              <PasswordField
                required
                value={form.password}
                onChange={(e) => update("password", e.target.value)}
                placeholder="Password (min. 8 characters)"
                minLength={8}
              />
            </div>
            <button type="submit" className="btn-primary w-full">
              Continue
            </button>
          </form>
        </>
      )}

      {step === 2 && (
        <>
          <h1 className="text-lg font-semibold text-ink-900">Verify the OTP</h1>
          <p className="text-sm text-slate-500 mt-1 mb-6">
            OTP has been sent to {form.email || "your registered email"}
          </p>
          <form onSubmit={finish} className="space-y-4">
            <div>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">OTP</label>
              <input
                required
                value={form.otp}
                onChange={(e) => update("otp", e.target.value)}
                placeholder="Enter 6-digit code"
                maxLength={6}
                className="input-field tracking-widest text-center"
              />
            </div>
            <button type="submit" disabled={loading} className="btn-primary w-full">
              {loading ? "Verifying..." : "Verify"}
            </button>
          </form>
        </>
      )}

      <p className="text-sm text-slate-500 text-center mt-6">
        Already have an account?{" "}
        <Link to="/login" className="text-brand-600 font-medium">
          Log in
        </Link>
      </p>
    </AuthLayout>
  );
}
