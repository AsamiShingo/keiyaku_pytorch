(this.webpackJsonpview=this.webpackJsonpview||[]).push([[0],{24:function(e,t,n){},25:function(e,t,n){},27:function(e,t,n){},46:function(e,t,n){},47:function(e,t,n){"use strict";n.r(t);var a=n(1),c=n.n(a),s=n(18),i=n.n(s),o=(n(24),n(19)),r=n(3),u=(n(25),n(0)),l=c.a.createContext(null);function d(){return c.a.useContext(l)}var j=function(e){var t=Object(a.useState)([]),n=Object(r.a)(t,2),c=n[0],s=n[1],i={messageList:c,setMessages:function(e){if(null===e||void 0===e)return;s(e)},addMessages:function(t){if(null===t||void 0===t)return;s((function(n){var a,c=Object(o.a)(t);try{for(c.s();!(a=c.n()).done;){var s=a.value;n.length>=e.maxMessageNum&&n.shift(),n=n.concat(s)}}catch(i){c.e(i)}finally{c.f()}return n}))},addMessage:function(t){s((function(n){return n.length>=e.maxMessageNum&&n.shift(),n.concat(t)}))},clearMessage:function(){s([])}};return Object(u.jsx)(l.Provider,{value:i,children:e.children})},f=function(){var e=d();return Object(u.jsx)("ul",{className:"MessageList",children:e.messageList.map((function(e,t){return Object(u.jsx)("li",{className:"message_"+e.category,children:e.message},t.toString())}))})},b=(n(9),n(27),n(4)),m=n.n(b),h=(n(46),function(e){var t=Object(a.useState)(!1),n=Object(r.a)(t,2),c=n[0],s=n[1],i=Object(a.useRef)(!1);return Object(a.useEffect)((function(){return i.current=!0,function(){i.current=!1}}),[]),Object(u.jsxs)("button",{className:"ui blue left floated button",onClick:function(t){s(!0);var n=new FormData;Object.keys(e.params).forEach((function(t){return n.append(t,e.params[t])})),m.a.post(e.url,n,{}).then((function(t){e.onSuccess(t),i.current&&s(!1)})).catch((function(t){e.onException(t),i.current&&s(!1)}))},children:[Object(u.jsx)("i",{className:"ui play icon "+(c?"loading":"")}),e.children]})}),p=function(e){var t=Object(a.useState)(!1),n=Object(r.a)(t,2),c=n[0],s=n[1];return Object(u.jsxs)("span",{children:[Object(u.jsx)("input",{type:"file",className:"FileUploadButton",id:"FileUploadButton",name:"FileUploadButton",onChange:function(t){if(null!==t.currentTarget.files){s(!0);var n=t.currentTarget.files[0],a=new FormData;a.append("file",n),m.a.post(e.url,a,{headers:{"content-type":"multipart/form-data"}}).then((function(t){e.onSuccess(t),s(!1)})).catch((function(t){e.onException(t),s(!1)}))}}}),Object(u.jsxs)("label",{htmlFor:"FileUploadButton",className:"ui blue left floated button",children:[Object(u.jsx)("i",{className:"ui upload icon "+(c?"loading":"")}),e.children]})]})},x=function(e){var t=Object(a.useState)(!1),n=Object(r.a)(t,2),c=n[0],s=n[1];return Object(u.jsxs)("button",{className:"ui blue left floated button",onClick:function(t){s(!0);var n=new FormData;Object.keys(e.params).forEach((function(t){return n.append(t,e.params[t])})),m.a.post(e.url,n,{responseType:"blob"}).then((function(t){var n=t.headers["content-type"],a=new Blob([t.data],{type:n}),c=window.URL.createObjectURL(a),i=document.createElement("a");i.href=c,i.setAttribute("download",e.filename),document.body.appendChild(i),i.click(),e.onSuccess(t),s(!1)})).catch((function(t){e.onException(t),s(!1)}))},children:[Object(u.jsx)("i",{className:"ui download icon "+(c?"loading":"")}),e.children]})},O=function(){var e=Object(a.useState)([]),t=Object(r.a)(e,2),n=t[0],c=t[1],s=d().addMessages;function i(e){return"ANALYZE_"+e.substr(0,e.lastIndexOf("."))+".txt"}function o(e){return function(t){!function(e){var t=n.filter((function(t){return t.seqid!==e}));c(t)}(e)}}function l(e){s([{category:"info",message:"testmessage1"},{category:"error",message:"testmessage2"}]),console.log("onSuccess")}function j(e){console.log("onException")}return Object(a.useEffect)((function(){m.a.get("/keiyaku_group/api/list").then((function(e){c(e.data.data),s(e.data.message)}))}),[]),Object(u.jsxs)("div",{children:[Object(u.jsx)(p,{url:"/keiyaku_group/api/upload",onSuccess:function(e){var t=e.data.code,n=e.data.data.seqid,a=e.data.data.filename;0===t&&function(e,t){c((function(n){return n.concat({seqid:e,filename:t})}))}(n,a),s(e.data.message)},onException:j,children:"\u5951\u7d04\u66f8\u30a2\u30c3\u30d7\u30ed\u30fc\u30c9"}),Object(u.jsxs)("table",{className:"ui selectable table",children:[Object(u.jsx)("thead",{children:Object(u.jsxs)("tr",{children:[Object(u.jsx)("th",{className:"th_no",children:"No"}),Object(u.jsx)("th",{className:"th_filename",children:"\u30d5\u30a1\u30a4\u30eb\u540d"}),Object(u.jsx)("th",{className:"th_button"})]})}),Object(u.jsx)("tbody",{children:n.map((function(e,t){return Object(u.jsxs)("tr",{children:[Object(u.jsx)("td",{className:"td_no",children:t+1}),Object(u.jsx)("td",{className:"td_filename",children:e.filename}),Object(u.jsxs)("td",{className:"td_button",children:[Object(u.jsx)(x,{url:"/keiyaku_group/api/download",filename:e.filename,params:{seqid:e.seqid},onSuccess:l,onException:j,children:"\u30c0\u30a6\u30f3\u30ed\u30fc\u30c9"}),Object(u.jsx)(x,{url:"/keiyaku_group/api/download_txt",filename:(n=e.filename,"TEXT_"+n.substr(0,n.lastIndexOf("."))+".txt"),params:{seqid:e.seqid},onSuccess:l,onException:j,children:"\u30c0\u30a6\u30f3\u30ed\u30fc\u30c9(txt)"}),Object(u.jsx)(x,{url:"/keiyaku_group/api/analyze",filename:i(e.filename),params:{seqid:e.seqid},onSuccess:l,onException:j,children:"\u89e3\u6790"}),Object(u.jsx)(h,{url:"/keiyaku_group/api/delete",params:{seqid:e.seqid},onSuccess:o(e.seqid),onException:j,children:"\u524a\u9664"})]})]},e.seqid);var n}))})]})]})};i.a.render(Object(u.jsxs)(c.a.StrictMode,{children:[Object(u.jsx)("div",{className:"ui container",children:Object(u.jsx)("h1",{children:"\u5951\u7d04\u6587\u7ae0\u89e3\u6790"})}),Object(u.jsxs)(j,{maxMessageNum:5,children:[Object(u.jsx)("div",{className:"ui container",children:Object(u.jsx)(O,{})}),Object(u.jsx)("div",{className:"ui container",children:Object(u.jsx)(f,{})})]})]}),document.getElementById("root"))}},[[47,1,2]]]);
//# sourceMappingURL=main.2b9f9a32.chunk.js.map