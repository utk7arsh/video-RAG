
import React from 'react';
import { useTheme } from '@/contexts/ThemeContext';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { VideoUploader } from '@/components/upload/VideoUploader';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

const Upload = () => {
  const { toggleTheme, isDarkMode } = useTheme();
  
  return (
    <DashboardLayout toggleTheme={toggleTheme} isDarkMode={isDarkMode}>
      <div className="p-8">
        <h1 className="text-3xl font-bold mb-6">Upload Videos</h1>
        
        <div className="grid grid-cols-1 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Upload New Video</CardTitle>
            </CardHeader>
            <CardContent>
              <VideoUploader />
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle>Upload Tips</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="list-disc pl-5 space-y-2">
                <li>Maximum file size is 1GB per video</li>
                <li>Supported formats: MP4, MOV, AVI, and WebM</li>
                <li>For best results, upload videos with clear audio</li>
                <li>Processing time depends on video length and quality</li>
                <li>You'll receive a notification when analysis is complete</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default Upload;
