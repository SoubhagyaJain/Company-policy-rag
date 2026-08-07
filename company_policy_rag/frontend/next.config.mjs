/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: process.env.BACKEND_API_URL 
          ? `${process.env.BACKEND_API_URL}/api/:path*` 
          : 'http://localhost:8000/api/:path*',
      },
    ];
  },
};

export default nextConfig;
