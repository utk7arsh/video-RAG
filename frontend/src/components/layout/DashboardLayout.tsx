
import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { 
  Moon, 
  Sun, 
  Upload, 
  Video, 
  Book, 
  Settings, 
  User,
  Search,
  Play
} from 'lucide-react';

interface DashboardLayoutProps {
  children: React.ReactNode;
  toggleTheme: () => void;
  isDarkMode: boolean;
}

export const DashboardLayout = ({ 
  children,
  toggleTheme,
  isDarkMode
}: DashboardLayoutProps) => {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const location = useLocation();
  
  const navItems = [
    { name: 'Videos', path: '/dashboard', icon: Video },
    { name: 'Upload', path: '/dashboard/upload', icon: Upload },
    { name: 'Playground', path: '/playground', icon: Play },
    { name: 'Analysis', path: '/dashboard/analysis', icon: Search },
    { name: 'Library', path: '/dashboard/library', icon: Book },
    { name: 'Settings', path: '/dashboard/settings', icon: Settings },
  ];
  
  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className={`${isSidebarCollapsed ? 'w-16' : 'w-64'} transition-all duration-300 bg-sidebar border-r border-border h-screen fixed left-0 top-0 z-30`}>
        <div className="flex flex-col h-full">
          {/* Logo/Header */}
          <div className="p-4 border-b border-border flex items-center justify-between">
            {!isSidebarCollapsed && (
              <Link to="/dashboard" className="flex items-center space-x-2">
                <div className="w-8 h-8 rounded-full gradient-bg flex items-center justify-center">
                  <span className="text-white font-bold text-sm">VI</span>
                </div>
                <span className="font-bold">VideoInsight</span>
              </Link>
            )}
            {isSidebarCollapsed && (
              <Link to="/dashboard" className="mx-auto">
                <div className="w-8 h-8 rounded-full gradient-bg flex items-center justify-center">
                  <span className="text-white font-bold text-sm">VI</span>
                </div>
              </Link>
            )}
          </div>

          {/* Navigation */}
          <nav className="flex-1 overflow-y-auto py-4">
            <ul className="space-y-2 px-2">
              {navItems.map((item) => (
                <li key={item.name}>
                  <Link 
                    to={item.path}
                    className={`flex items-center rounded-md py-2 px-3 transition-colors ${
                      location.pathname === item.path 
                        ? 'bg-primary text-primary-foreground' 
                        : 'hover:bg-sidebar-accent'
                    }`}
                  >
                    <item.icon className={`h-5 w-5 ${isSidebarCollapsed ? 'mx-auto' : 'mr-3'}`} />
                    {!isSidebarCollapsed && <span>{item.name}</span>}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>

          {/* Footer */}
          <div className="p-4 border-t border-border space-y-2">
            <Button 
              variant="ghost" 
              size="sm"
              className={`${isSidebarCollapsed ? 'w-full justify-center' : ''}`}
              onClick={toggleTheme}
            >
              {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              {!isSidebarCollapsed && <span className="ml-2">Theme</span>}
            </Button>

            <Link 
              to="/dashboard/profile" 
              className={`flex items-center space-x-3 rounded-md py-2 px-3 hover:bg-sidebar-accent transition-colors ${
                location.pathname === '/dashboard/profile' ? 'bg-sidebar-accent' : ''
              }`}
            >
              <User className={`h-5 w-5 ${isSidebarCollapsed ? 'mx-auto' : ''}`} />
              {!isSidebarCollapsed && <span>Profile</span>}
            </Link>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className={`flex-1 ${isSidebarCollapsed ? 'ml-16' : 'ml-64'} transition-all duration-300`}>
        {children}
      </main>
    </div>
  );
};
