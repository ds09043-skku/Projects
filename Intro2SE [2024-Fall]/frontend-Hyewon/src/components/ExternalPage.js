import React from 'react';

const ExternalPage = ({ url }) => {
  return (
    <div style={{ width: '100%', height: '100%', overflow: 'scroll' }}>
      <iframe
        src={url}
        style={{
          border: 'none',
          width: '100%',
          height: '100%',
        }}
        title="External Page"
      />
    </div>
  );
};

export default ExternalPage;
