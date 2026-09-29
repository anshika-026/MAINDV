import { Component } from "react";

// Catches a render error in one page so it shows a message and a reload
// button instead of blanking the whole app. Keyed by route in App.jsx, so
// navigating to another page clears the error.
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // eslint-disable-next-line no-console
    console.error("Page crashed:", error, info?.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="p-8 max-w-lg mx-auto text-center">
        <h1 className="text-lg font-semibold text-ink-900">Something went wrong on this page</h1>
        <p className="text-sm text-slate-500 mt-2">
          The rest of the app is still working. Reload to try again; if it keeps happening, tell your administrator.
        </p>
        <button className="btn-primary mt-4" onClick={() => window.location.reload()}>
          Reload
        </button>
      </div>
    );
  }
}
