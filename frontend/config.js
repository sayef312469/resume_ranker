// API Configuration
// Automatically uses localhost:8000 for dev, and deployment domain for production

const API_BASE = (typeof window !== 'undefined' && window.location.hostname === 'localhost')
  ? 'http://localhost:8000'
  : `${window.location.protocol}//${window.location.host}`;

