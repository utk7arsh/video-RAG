
import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Send, User, Bot } from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface ChatInterfaceProps {
  videoUrl: string | null;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ videoUrl }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Add initial assistant message
  useEffect(() => {
    if (videoUrl) {
      setMessages([
        {
          id: '1',
          role: 'assistant',
          content: 'I\'ve analyzed your video. What would you like to know about it?',
          timestamp: new Date(),
        },
      ]);
    } else {
      setMessages([
        {
          id: '1',
          role: 'assistant',
          content: 'Please upload a video to start the conversation.',
          timestamp: new Date(),
        },
      ]);
    }
  }, [videoUrl]);
  
  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  const handleSendMessage = () => {
    if (input.trim() === '') return;
    
    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    
    // Simulate AI response after a delay
    setTimeout(() => {
      // Mock AI response based on the question
      let response = "I'm analyzing the video to answer your question...";
      
      if (input.toLowerCase().includes('summary')) {
        response = "This video appears to be an educational lecture about data science concepts, specifically focusing on machine learning algorithms and their applications in real-world scenarios.";
      } else if (input.toLowerCase().includes('who')) {
        response = "The presenter in this video is Dr. Jane Smith, an expert in the field of AI and machine learning with over 10 years of experience in both academia and industry.";
      } else if (input.toLowerCase().includes('when') || input.toLowerCase().includes('time')) {
        response = "This concept is discussed at approximately 4:25 in the video, where the lecturer begins explaining the core principles.";
      } else {
        response = "Based on my analysis of the video, the question you're asking relates to the concepts presented around the middle section of the lecture. The speaker explains that machine learning models require proper training data to make accurate predictions, which addresses your query about model accuracy.";
      }
      
      const assistantMessage: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        content: response,
        timestamp: new Date(),
      };
      
      setMessages(prev => [...prev, assistantMessage]);
      setIsLoading(false);
    }, 1500);
  };
  
  return (
    <div className="flex flex-col h-full">
      {/* Chat messages area */}
      <ScrollArea className="flex-1 p-4 mb-4 rounded-md border">
        <div className="space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`flex items-start space-x-2 max-w-[80%] ${
                  message.role === 'user'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted'
                } p-3 rounded-lg`}
              >
                <div className="w-8 h-8 rounded-full flex items-center justify-center bg-background">
                  {message.role === 'user' ? (
                    <User className="h-4 w-4" />
                  ) : (
                    <Bot className="h-4 w-4" />
                  )}
                </div>
                <div className="flex-1">
                  <p className="text-sm">{message.content}</p>
                </div>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-muted p-3 rounded-lg">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 rounded-full bg-foreground/40 animate-pulse"></div>
                  <div className="w-2 h-2 rounded-full bg-foreground/40 animate-pulse delay-150"></div>
                  <div className="w-2 h-2 rounded-full bg-foreground/40 animate-pulse delay-300"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>
      
      {/* Input area */}
      <div className="flex gap-2">
        <Textarea
          placeholder={videoUrl ? "Ask a question about the video..." : "Upload a video first..."}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="resize-none"
          disabled={!videoUrl || isLoading}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSendMessage();
            }
          }}
        />
        <Button 
          onClick={handleSendMessage} 
          disabled={!videoUrl || isLoading || input.trim() === ''}
          size="icon"
          className="self-end"
        >
          <Send className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
};
