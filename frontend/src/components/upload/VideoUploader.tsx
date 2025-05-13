
import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, Video } from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';

interface VideoUploaderProps {
  onVideoUploaded?: (videoUrl: string) => void;
}

export const VideoUploader: React.FC<VideoUploaderProps> = ({ onVideoUploaded }) => {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const videoFiles = acceptedFiles.filter(file => file.type.startsWith('video/'));
    setFiles(prev => [...prev, ...videoFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': []
    },
    maxSize: 1000 * 1024 * 1024 // 1GB max
  });

  const handleRemoveFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = () => {
    if (files.length === 0) return;
    
    setUploading(true);
    
    // Simulate upload progress
    let progress = 0;
    const interval = setInterval(() => {
      progress += 5;
      setUploadProgress(progress);
      
      if (progress >= 100) {
        clearInterval(interval);
        setUploading(false);
        
        // Create a temporary URL for the uploaded video
        if (files[0] && onVideoUploaded) {
          const videoUrl = URL.createObjectURL(files[0]);
          onVideoUploaded(videoUrl);
        }
        
        console.log('Upload completed');
      }
    }, 300);
  };

  return (
    <div className="w-full space-y-6">
      {/* Dropzone area */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-primary bg-primary/10' : 'border-border hover:border-primary/50'}
          ${uploading ? 'pointer-events-none opacity-50' : ''}
        `}
      >
        <input {...getInputProps()} disabled={uploading} />
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
            <Upload className="h-8 w-8 text-primary" />
          </div>
          <div>
            <p className="text-lg font-medium">Drag & drop video files here</p>
            <p className="text-sm text-muted-foreground mt-1">
              or click to browse (max 1GB)
            </p>
          </div>
        </div>
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div className="space-y-4">
          <h3 className="font-semibold text-lg">Selected Files</h3>
          
          <div className="space-y-3">
            {files.map((file, index) => (
              <div 
                key={`${file.name}-${index}`} 
                className="flex items-center justify-between p-3 border rounded-md bg-background"
              >
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 rounded bg-primary/10 flex items-center justify-center">
                    <Video className="h-5 w-5 text-primary" />
                  </div>
                  <div className="flex flex-col">
                    <span className="font-medium truncate max-w-[200px] sm:max-w-xs">{file.name}</span>
                    <span className="text-xs text-muted-foreground">{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                  </div>
                </div>
                <Button 
                  variant="ghost" 
                  size="icon"
                  onClick={() => handleRemoveFile(index)}
                  disabled={uploading}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>

          {/* Upload progress */}
          {uploading && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Uploading...</span>
                <span>{uploadProgress}%</span>
              </div>
              <Progress value={uploadProgress} />
            </div>
          )}

          {/* Upload button */}
          <div className="flex justify-end">
            <Button 
              onClick={handleUpload} 
              disabled={uploading || files.length === 0}
              className={uploading ? '' : 'gradient-bg'}
            >
              {uploading ? 'Uploading...' : 'Upload Files'}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};
