
import React, { useState, useEffect } from "react";
import 'semantic-ui-css/semantic.min.css'
import './KeiyakuTable.css'
import axios, { AxiosResponse, AxiosError } from 'axios'
import { PostButton, FileUploadButton, FileDownloadButton } from "./CommonButton"
import { useMessageListContext } from "./CommonMessageList";

const KEIYAKU_GROUP_DOMAIN = process.env.NODE_ENV === "development" ? "http://localhost" : "";

type KeiyakuRecordType = {
  seqid: string;
  filename: string;
}

export const KeiyakuTable = () => {
  const [keiyakuRecords, setKeiyakuRecords] = useState<KeiyakuRecordType[]>([]);
  const { addMessages } = useMessageListContext();

  useEffect(() => {
    axios.get(KEIYAKU_GROUP_DOMAIN + '/keiyaku_group/api/list')
      .then((response: AxiosResponse) => {
        setKeiyakuRecords(response.data["data"])
        addMessages(response.data["message"])
      })
  },[])

  function addRecord(seqid: string, filename: string) {
    setKeiyakuRecords((prev) => prev.concat({seqid: seqid, filename: filename }));
  }

  function deleteRecord(seqid: string) {
    const records = keiyakuRecords.filter((record) => record.seqid !== seqid);
    setKeiyakuRecords(records);
  }

  function getAnalyzeFileName(orgfilename: string) {
    return "ANALYZE_" + orgfilename.substr(0, orgfilename.lastIndexOf('.')) + ".txt";
  }

  function getTextFileName(orgfilename: string) {
    return "TEXT_" + orgfilename.substr(0, orgfilename.lastIndexOf('.')) + ".txt";
  }

  function onFileUploadSuccess() {
    return (response: AxiosResponse) => {
      const code = response.data["code"];
      const seqid = response.data["data"]["seqid"];
      const filename = response.data["data"]["filename"];
      
      if(code === 0) {
        addRecord(seqid, filename); 
      }

      addMessages(response.data["message"])
    }
  }

  function onDeleteSuccess(seqid: string) {
    return (response: AxiosResponse) => {
      deleteRecord(seqid);
    }
  }

  function onSuccess(response: AxiosResponse) {
    addMessages([{ category: "info", message: "testmessage1" }, { category: "error", message: "testmessage2" }]);
    console.log("onSuccess");
  }

  function onException(exception: AxiosError) {
    console.log("onException");
  }

  return (
		<div> 
      <FileUploadButton url={ KEIYAKU_GROUP_DOMAIN + '/keiyaku_group/api/upload' } onSuccess={onFileUploadSuccess()} onException={onException}>契約書アップロード</FileUploadButton>
			<table className="ui selectable table">
			<thead>
			<tr>
				<th className="th_no">No</th>
				<th className="th_filename">ファイル名</th>
				<th className="th_button"></th>
			</tr>
			</thead>			
			<tbody>
        { keiyakuRecords.map((record, index) =>
          <tr key={record.seqid}>
            <td className="td_no">{index+1}</td>
            <td className="td_filename">{record.filename}</td>
            <td className="td_button">
              <FileDownloadButton url={ KEIYAKU_GROUP_DOMAIN + '/keiyaku_group/api/download' } filename={record.filename} params={{"seqid": record.seqid}}
                onSuccess={onSuccess} onException={onException}>ダウンロード</FileDownloadButton>
              <FileDownloadButton url={ KEIYAKU_GROUP_DOMAIN + '/keiyaku_group/api/download_txt' } filename={getTextFileName(record.filename)} params={{"seqid": record.seqid}}
                onSuccess={onSuccess} onException={onException}>ダウンロード(txt)</FileDownloadButton>
              <FileDownloadButton url={ KEIYAKU_GROUP_DOMAIN + '/keiyaku_group/api/analyze' } filename={getAnalyzeFileName(record.filename)} params={{"seqid": record.seqid}}
                onSuccess={onSuccess} onException={onException}>解析</FileDownloadButton>
              <PostButton url={ KEIYAKU_GROUP_DOMAIN + '/keiyaku_group/api/delete' } params={{"seqid": record.seqid}}
                onSuccess={onDeleteSuccess(record.seqid)} onException={onException}>削除</PostButton>
            </td>
          </tr>
        )}
      </tbody>
			</table>			
		</div>
  );
}