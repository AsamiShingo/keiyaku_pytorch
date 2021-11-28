import React, { useState, ReactNode } from "react";
import './CommonMessageList.css'

type Props = {
    children: ReactNode,
    maxMessageNum: number
}

type MessageType = {
    category: string,
    message: string
}

type MessageListContextType = {
    messageList: MessageType[],
    addMessages: (messages: MessageType[]) => void,
    setMessages: (messages: MessageType[]) => void,
    addMessage: (message: MessageType) => void,
    clearMessage: () => void
};

const messageListContext = React.createContext<MessageListContextType | null>(null);

export function useMessageListContext() {
  return React.useContext(messageListContext)!;
}

export const MessageListProvider = (props: Props) => {
    const [messageList, setMessageList] = useState<MessageType[]>([]);
    const provider: MessageListContextType = {
        messageList: messageList,
        setMessages: setMessages,
        addMessages: addMessages,
        addMessage: addMessage,
        clearMessage: clearMessage,
    };

    function setMessages(messages: MessageType[]) {
        if(messages === null || messages === undefined) {
            return;
        }
        
        setMessageList(messages);
    }

    function addMessages(messages: MessageType[]) {
        if(messages === null || messages === undefined) {
            return;
        }

        setMessageList((prev) => {
            for(const message of messages) {
                if(prev.length >= props.maxMessageNum) {
                    prev.shift();
                }
                prev = prev.concat(message);
            }

            return prev;
        });
    }

    function addMessage(message: MessageType) {
        setMessageList((prev) => {
            if(prev.length >= props.maxMessageNum) {
                prev.shift();
            }
            return prev.concat(message);
        });
    }
  
    function clearMessage() {
        setMessageList([]);
    }
    
    return (
        <messageListContext.Provider value={provider}>
            { props.children }
        </messageListContext.Provider>
    )
}

export const MessageList = () => {
    const context = useMessageListContext();
    
    return (
        <ul className="MessageList">
            { context.messageList.map((message, index) =>
                <li className={ "message_" + message.category } key={index.toString()}>{ message.message }</li>
            )}
        </ul>
    )
}