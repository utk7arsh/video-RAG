
import React from 'react';
import { useTheme } from '@/contexts/ThemeContext';
import { DashboardLayout } from '@/components/layout/DashboardLayout';

const Dashboard = () => {
  const { toggleTheme, isDarkMode } = useTheme();
  
  return (
    <DashboardLayout toggleTheme={toggleTheme} isDarkMode={isDarkMode}>
      <div className="p-8">
        <h1 className="text-3xl font-bold mb-6">Dashboard</h1>
        <div className="grid grid-cols-1 gap-6">
          {/* Content will be added here */}
          <div className="p-6 bg-card rounded-lg border border-border">
            <p className="text-muted-foreground">Your video dashboard will appear here.</p>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default Dashboard;
