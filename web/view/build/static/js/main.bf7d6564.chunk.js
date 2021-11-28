(this.webpackJsonpview=this.webpackJsonpview||[]).push([[0],{24:function(e,t,n){},25:function(e,t,n){},27:function(e,t,n){},46:function(e,t,n){},47:function(e,t,n){"use strict";n.r(t);var c=n(1),a=n.n(c),s=n(18),i=n.n(s),o=(n(24),n(19)),r=n(3),u=(n(25),n(0)),l=a.a.createContext(null);function d(){return a.a.useContext(l)}var f=function(e){var t=Object(c.useState)([]),n=Object(r.a)(t,2),a=n[0],s=n[1],i={messageList:a,setMessages:function(e){if(null===e||void 0===e)return;s(e)},addMessages:function(t){if(null===t||void 0===t)return;s((function(n){var c,a=Object(o.a)(t);try{for(a.s();!(c=a.n()).done;){var s=c.value;n.length>=e.maxMessageNum&&n.shift(),n=n.concat(s)}}catch(i){a.e(i)}finally{a.f()}return n}))},addMessage:function(t){s((function(n){return n.length>=e.maxMessageNum&&n.shift(),n.concat(t)}))},clearMessage:function(){s([])}};return Object(u.jsx)(l.Provider,{value:i,children:e.children})},j=function(){var e=d();return Object(u.jsx)("ul",{className:"MessageList",children:e.messageList.map((function(e,t){return Object(u.jsx)("li",{className:"message_"+e.category,children:e.message},t.toString())}))})},b=(n(9),n(27),n(4)),m=n.n(b),p=(n(46),function(e){var t=Object(c.useState)(!1),n=Object(r.a)(t,2),a=n[0],s=n[1],i=Object(c.useRef)(!1);return Object(c.useEffect)((function(){return i.current=!0,function(){i.current=!1}}),[]),Object(u.jsxs)("button",{className:"ui blue left floated button",onClick:function(t){s(!0);var n=new FormData;Object.keys(e.params).forEach((function(t){return n.append(t,e.params[t])})),m.a.post(e.url,n,{}).then((function(t){void 0!==e.onSuccess&&e.onSuccess(t),i.current&&s(!1)})).catch((function(t){void 0!==e.onException&&e.onException(t),i.current&&s(!1)}))},children:[Object(u.jsx)("i",{className:"ui play icon "+(a?"loading":"")}),e.children]})}),h=function(e){var t=Object(c.useState)(!1),n=Object(r.a)(t,2),a=n[0],s=n[1];return Object(u.jsxs)("span",{children:[Object(u.jsx)("input",{type:"file",className:"FileUploadButton",id:"FileUploadButton",name:"FileUploadButton",onChange:function(t){if(null!==t.currentTarget.files){s(!0);var n=t.currentTarget.files[0],c=new FormData;c.append("file",n),m.a.post(e.url,c,{headers:{"content-type":"multipart/form-data"}}).then((function(t){void 0!==e.onSuccess&&e.onSuccess(t),s(!1)})).catch((function(t){void 0!==e.onException&&e.onException(t),s(!1)}))}}}),Object(u.jsxs)("label",{htmlFor:"FileUploadButton",className:"ui blue left floated button",children:[Object(u.jsx)("i",{className:"ui upload icon "+(a?"loading":"")}),e.children]})]})},x=function(e){var t=Object(c.useState)(!1),n=Object(r.a)(t,2),a=n[0],s=n[1];return Object(u.jsxs)("button",{className:"ui blue left floated button",onClick:function(t){s(!0);var n=new FormData;Object.keys(e.params).forEach((function(t){return n.append(t,e.params[t])})),m.a.post(e.url,n,{responseType:"blob"}).then((function(t){var n=t.headers["content-type"],c=new Blob([t.data],{type:n}),a=window.URL.createObjectURL(c),i=document.createElement("a");i.href=a,i.setAttribute("download",e.filename),document.body.appendChild(i),i.click(),void 0!==e.onSuccess&&e.onSuccess(t),s(!1)})).catch((function(t){void 0!==e.onException&&e.onException(t),s(!1)}))},children:[Object(u.jsx)("i",{className:"ui download icon "+(a?"loading":"")}),e.children]})},O=function(){var e=Object(c.useState)([]),t=Object(r.a)(e,2),n=t[0],a=t[1],s=d(),i=s.addMessage,o=s.addMessages;function l(e){return"ANALYZE_"+e.substr(0,e.lastIndexOf("."))+".txt"}function f(e){return function(t){!function(e){var t=n.filter((function(t){return t.seqid!==e}));a(t)}(e)}}function j(e){}function b(e){return function(t){i({category:"error",message:e})}}return Object(c.useEffect)((function(){m.a.get("/keiyaku_group/api/list").then((function(e){a(e.data.data),o(e.data.message)}))}),[]),Object(u.jsxs)("div",{children:[Object(u.jsx)(h,{url:"/keiyaku_group/api/upload",onSuccess:function(e){var t=e.data.code,n=e.data.data.seqid,c=e.data.data.filename;0===t&&function(e,t){a((function(n){return n.concat({seqid:e,filename:t})}))}(n,c),o(e.data.message)},onException:b("\u5951\u7d04\u66f8\u306e\u30a2\u30c3\u30d7\u30ed\u30fc\u30c9\u306b\u5931\u6557\u3057\u307e\u3057\u305f"),children:"\u5951\u7d04\u66f8\u30a2\u30c3\u30d7\u30ed\u30fc\u30c9"}),Object(u.jsxs)("table",{className:"ui selectable table",children:[Object(u.jsx)("thead",{children:Object(u.jsxs)("tr",{children:[Object(u.jsx)("th",{className:"th_no",children:"No"}),Object(u.jsx)("th",{className:"th_filename",children:"\u30d5\u30a1\u30a4\u30eb\u540d"}),Object(u.jsx)("th",{className:"th_button"})]})}),Object(u.jsx)("tbody",{children:n.map((function(e,t){return Object(u.jsxs)("tr",{children:[Object(u.jsx)("td",{className:"td_no",children:t+1}),Object(u.jsx)("td",{className:"td_filename",children:e.filename}),Object(u.jsxs)("td",{className:"td_button",children:[Object(u.jsx)(x,{url:"/keiyaku_group/api/download",filename:e.filename,params:{seqid:e.seqid},onSuccess:j,onException:b(e.filename+"\u306e\u30c0\u30a6\u30f3\u30ed\u30fc\u30c9\u306b\u5931\u6557\u3057\u307e\u3057\u305f"),children:"\u30c0\u30a6\u30f3\u30ed\u30fc\u30c9"}),Object(u.jsx)(x,{url:"/keiyaku_group/api/download_txt",filename:(n=e.filename,"TEXT_"+n.substr(0,n.lastIndexOf("."))+".txt"),params:{seqid:e.seqid},onSuccess:j,onException:b(e.filename+"\u306e\u30c0\u30a6\u30f3\u30ed\u30fc\u30c9(txt)\u306b\u5931\u6557\u3057\u307e\u3057\u305f"),children:"\u30c0\u30a6\u30f3\u30ed\u30fc\u30c9(txt)"}),Object(u.jsx)(x,{url:"/keiyaku_group/api/analyze",filename:l(e.filename),params:{seqid:e.seqid},onSuccess:j,onException:b(e.filename+"\u306e\u89e3\u6790\u306b\u5931\u6557\u3057\u307e\u3057\u305f"),children:"\u89e3\u6790"}),Object(u.jsx)(p,{url:"/keiyaku_group/api/delete",params:{seqid:e.seqid},onSuccess:f(e.seqid),onException:b(e.filename+"\u306e\u524a\u9664\u306b\u5931\u6557\u3057\u307e\u3057\u305f"),children:"\u524a\u9664"})]})]},e.seqid);var n}))})]})]})};i.a.render(Object(u.jsxs)(a.a.StrictMode,{children:[Object(u.jsx)("div",{className:"ui container",children:Object(u.jsx)("h1",{children:"\u5951\u7d04\u6587\u7ae0\u89e3\u6790"})}),Object(u.jsxs)(f,{maxMessageNum:5,children:[Object(u.jsx)("div",{className:"ui container",children:Object(u.jsx)(O,{})}),Object(u.jsx)("div",{className:"ui container",children:Object(u.jsx)(j,{})})]})]}),document.getElementById("root"))}},[[47,1,2]]]);
//# sourceMappingURL=main.bf7d6564.chunk.js.map