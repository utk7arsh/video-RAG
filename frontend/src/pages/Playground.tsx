
import React, { useState } from 'react';
import { useTheme } from '@/contexts/ThemeContext';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { VideoUploader } from '@/components/upload/VideoUploader';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ChatInterface } from '@/components/playground/ChatInterface';
import { VideoPlayer } from '@/components/playground/VideoPlayer';
import { ClipGenerator } from '@/components/playground/ClipGenerator';

const Playground = () => {
  const { toggleTheme, isDarkMode } = useTheme();
  const [currentVideo, setCurrentVideo] = useState<string | null>(null);
  const [videoUploaded, setVideoUploaded] = useState(false);
  
  const handleVideoUpload = (videoUrl: string) => {
    setCurrentVideo(videoUrl);
    setVideoUploaded(true);
  };
  
  return (
    <DashboardLayout toggleTheme={toggleTheme} isDarkMode={isDarkMode}>
      <div className="p-4 md:p-8">
        <h1 className="text-3xl font-bold mb-6">Video Playground</h1>
        
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left side: Video upload and player */}
          <div className="lg:col-span-7">
            <Card className="h-full">
              <CardHeader>
                <CardTitle>Video</CardTitle>
                <CardDescription>
                  Upload a video or try one of our sample videos
                </CardDescription>
              </CardHeader>
              <CardContent>
                {!videoUploaded ? (
                  <VideoUploader onVideoUploaded={handleVideoUpload} />
                ) : (
                  <VideoPlayer videoUrl={currentVideo} />
                )}
              </CardContent>
            </Card>
          </div>
          
          {/* Right side: Chat and clip generation tabs */}
          <div className="lg:col-span-5">
            <Tabs defaultValue="chat" className="h-full flex flex-col">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="chat">Ask Questions</TabsTrigger>
                <TabsTrigger value="clips">Generate Clips</TabsTrigger>
              </TabsList>
              
              <div className="flex-1 overflow-hidden">
                <TabsContent value="chat" className="h-full">
                  <Card className="h-full">
                    <CardHeader>
                      <CardTitle>Chat with Your Video</CardTitle>
                      <CardDescription>
                        Ask questions about the video content
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="h-[calc(100%-5rem)]">
                      <ChatInterface videoUrl={currentVideo} />
                    </CardContent>
                  </Card>
                </TabsContent>
                
                <TabsContent value="clips" className="h-full">
                  <Card className="h-full">
                    <CardHeader>
                      <CardTitle>Generate Clips</CardTitle>
                      <CardDescription>
                        Create clips of key moments from the video
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="h-[calc(100%-5rem)]">
                      <ClipGenerator videoUrl={currentVideo} />
                    </CardContent>
                  </Card>
                </TabsContent>
              </div>
            </Tabs>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default Playground;
