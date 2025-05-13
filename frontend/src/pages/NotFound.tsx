
import React from 'react';
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';

const NotFound = () => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="text-center p-4">
        <h1 className="text-9xl font-bold gradient-text">404</h1>
        <h2 className="text-3xl font-semibold mt-4 mb-2">Page Not Found</h2>
        <p className="text-muted-foreground mb-8">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <Link to="/">
          <Button className="gradient-bg">Return to Home</Button>
        </Link>
      </div>
    </div>
  );
};

export default NotFound;
