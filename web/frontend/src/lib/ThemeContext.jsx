import React, { createContext, useContext, useEffect, useState } from 'react';

const ThemeContext = createContext(null);
const STORAGE_KEY = 'topdt-theme';
const THEMES = ['light', 'dark', 'warm'];

function getInitialTheme() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (THEMES.includes(stored)) return stored;
  } catch {
    // localStorage unavailable (private mode, etc.) -- fall through to default
  }
  return 'light';
}

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(getInitialTheme);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      // ignore write failures
    }
  }, [theme]);

  return <ThemeContext.Provider value={{ theme, setTheme, THEMES }}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme must be used within a ThemeProvider');
  return ctx;
}
