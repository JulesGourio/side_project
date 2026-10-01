export interface AppBranding {
  name: string;
  logo: string;
  company_name?: string;
}

export interface AppConfig {
  app_name: string;
  branding: AppBranding;
}

export interface UserMe {
  user: string;
  workspace_url: string;
}

let cachedConfig: AppConfig | null = null;
let cachedMe: UserMe | null = null;

export async function getAppConfig(): Promise<AppConfig> {
  if (cachedConfig) return cachedConfig;
  try {
    const res = await fetch('/api/config/app');
    if (!res.ok) throw new Error(res.statusText);
    cachedConfig = await res.json();
    return cachedConfig!;
  } catch {
    return {
      app_name: 'LatLang',
      branding: { name: 'LatLang', logo: '/logos/LOGO_LATECOERE.png' },
    };
  }
}

export async function getUserMe(): Promise<UserMe> {
  if (cachedMe) return cachedMe;
  try {
    const res = await fetch('/api/me');
    if (!res.ok) throw new Error(res.statusText);
    cachedMe = await res.json();
    return cachedMe!;
  } catch {
    return { user: '', workspace_url: '' };
  }
}
