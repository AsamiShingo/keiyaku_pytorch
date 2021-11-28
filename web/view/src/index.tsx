import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import { MessageListProvider, MessageList } from './CommonMessageList';
import { KeiyakuTable } from './KeiyakuTable';

ReactDOM.render(
  <React.StrictMode>
    <div className="ui container">
      <h1>契約文章解析</h1>
    </div>    
    <MessageListProvider maxMessageNum={5}>
      <div className="ui container">
        <KeiyakuTable />
      </div>
      <div className="ui container">
        <MessageList />
      </div>
    </MessageListProvider>
  </React.StrictMode>,
  document.getElementById('root')
);