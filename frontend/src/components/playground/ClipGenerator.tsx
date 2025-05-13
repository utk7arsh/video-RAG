
import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Scissors, Download, Search } from 'lucide-react';

interface Clip {
  id: string;
  title: string;
  startTime: number;
  endTime: number;
  thumbnail: string;
}

interface ClipGeneratorProps {
  videoUrl: string | null;
}

export const ClipGenerator: React.FC<ClipGeneratorProps> = ({ videoUrl }) => {
  const [query, setQuery] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [clips, setClips] = useState<Clip[]>([]);
  
  const handleGenerateClips = () => {
    if (!query.trim() || !videoUrl) return;
    
    setIsGenerating(true);
    
    // Simulate clip generation with a delay
    setTimeout(() => {
      // Generate mock clips based on the query
      const newClips = [
        {
          id: `clip-${Date.now()}-1`,
          title: `Key concept: ${query}`,
          startTime: 125,
          endTime: 185,
          thumbnail: 'https://placehold.co/480x270/333/white?text=Clip+1',
        },
        {
          id: `clip-${Date.now()}-2`,
          title: `Example of ${query}`,
          startTime: 320,
          endTime: 350,
          thumbnail: 'https://placehold.co/480x270/333/white?text=Clip+2',
        },
        {
          id: `clip-${Date.now()}-3`,
          title: `Summary of ${query}`,
          startTime: 450,
          endTime: 490,
          thumbnail: 'https://placehold.co/480x270/333/white?text=Clip+3',
        },
      ];
      
      setClips(newClips);
      setIsGenerating(false);
    }, 2000);
  };
  
  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
  };
  
  return (
    <div className="flex flex-col h-full">
      <div className="flex gap-2 mb-4">
        <Input
          placeholder="What clips would you like to generate?"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={!videoUrl || isGenerating}
        />
        <Button
          onClick={handleGenerateClips}
          disabled={!videoUrl || isGenerating || !query.trim()}
        >
          {isGenerating ? (
            <>
              <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
              Generating
            </>
          ) : (
            <>
              <Scissors className="mr-2 h-4 w-4" />
              Generate
            </>
          )}
        </Button>
      </div>
      
      {!videoUrl && (
        <div className="flex-1 flex items-center justify-center border rounded-md">
          <p className="text-muted-foreground">Upload a video to generate clips</p>
        </div>
      )}
      
      {videoUrl && clips.length === 0 && !isGenerating && (
        <div className="flex-1 flex flex-col items-center justify-center border rounded-md p-8">
          <Search className="h-12 w-12 text-muted-foreground mb-4" />
          <p className="text-center text-muted-foreground">
            Enter a topic or concept to generate relevant clips from your video
          </p>
        </div>
      )}
      
      {videoUrl && isGenerating && (
        <div className="flex-1 flex flex-col items-center justify-center border rounded-md">
          <div className="h-12 w-12 animate-spin rounded-full border-4 border-primary border-t-transparent mb-4"></div>
          <p className="text-muted-foreground">Analyzing video content...</p>
        </div>
      )}
      
      {clips.length > 0 && (
        <ScrollArea className="flex-1 pr-4">
          <div className="space-y-4">
            {clips.map((clip) => (
              <div
                key={clip.id}
                className="border rounded-lg overflow-hidden"
              >
                <img
                  src={clip.thumbnail}
                  alt={clip.title}
                  className="w-full h-32 object-cover"
                />
                <div className="p-3">
                  <h4 className="font-medium truncate">{clip.title}</h4>
                  <p className="text-sm text-muted-foreground mb-2">
                    {formatTime(clip.startTime)} - {formatTime(clip.endTime)}
                  </p>
                  <div className="flex justify-between">
                    <Button variant="outline" size="sm">
                      Play Clip
                    </Button>
                    <Button variant="outline" size="sm">
                      <Download className="h-4 w-4 mr-1" />
                      Save
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      )}
    </div>
  );
};
