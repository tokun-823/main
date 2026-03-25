import { Link, useLocation } from 'react-router-dom';

export const GlobalNav = () => {
  const location = useLocation();
  
  const navItems = [
    { path: '/anti-pick', label: 'アンチピック' },
    { path: '/map-pick', label: 'マップ攻略' },
    { path: '/hero-table', label: 'ヒーロー一覧表' },
  ];
  
  return (
    <nav className="bg-surface border-b border-border">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center space-x-2 text-xl font-bold">
            <span className="text-white">OW2</span>
            <span className="text-accent">STRATEGY</span>
          </Link>
          
          <div className="hidden md:flex space-x-8">
            {navItems.map(item => (
              <Link
                key={item.path}
                to={item.path}
                className={`text-sm font-medium transition-colors ${
                  location.pathname === item.path
                    ? 'text-accent border-b-2 border-accent pb-1'
                    : 'text-text-sub hover:text-white'
                }`}
              >
                {item.label}
              </Link>
            ))}
          </div>
          
          <button className="md:hidden text-white">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>
      </div>
    </nav>
  );
};
