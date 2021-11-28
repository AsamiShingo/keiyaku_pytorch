import React, { useState, ReactNode, useEffect, useRef, MutableRefObject } from "react";
import axios, { AxiosResponse, AxiosError } from 'axios'
import 'semantic-ui-css/semantic.min.css'
import './CommonButton.css'

type NormalButtonProps = {
    children: ReactNode,
    onClick: (event: React.MouseEvent<HTMLButtonElement, MouseEvent>) => void
}

type PostButtonProps = {
    children: ReactNode,
    url: string,
    params: { [key: string]: any },
    onSuccess: (response: AxiosResponse) => void,
    onException: (exception: AxiosError) => void,
}

type FileUploadButtonProps = {
    children: ReactNode,
    url: string,
    onSuccess: (response: AxiosResponse) => void,
    onException: (exception: AxiosError) => void,
}

type FileDownloadButtonProps = {
    children: ReactNode,
    url: string,
    filename: string,
    params: { [key: string]: any },
    onSuccess: (response: AxiosResponse) => void,
    onException: (exception: AxiosError) => void,
}

export const NormalButton = (props: NormalButtonProps) => {

    function onClickFunction(e: React.MouseEvent<HTMLButtonElement, MouseEvent>) {
        props.onClick(e);
    }

    return (
        <button className="ui blue button" onClick={onClickFunction}>{props.children}</button>
    )
}

export const PostButton = (props: PostButtonProps) => {
    const [ iconAnimetionFlg, setIconAnimetionFlg ] = useState(false);
    const mounted: MutableRefObject<boolean> = useRef(false);

    useEffect(() => {
        mounted.current = true;
        return () => { mounted.current = false; }
    }, [])

    function onClickFunction(e: React.MouseEvent<HTMLButtonElement, MouseEvent>) {
        setIconAnimetionFlg(true);
        const form = new FormData();
        Object.keys(props.params).forEach((key) => form.append(key, props.params[key]));
        axios.post(props.url, form, {
        }).then((response: AxiosResponse) => {
            props.onSuccess(response);
            if(mounted.current) {
                setIconAnimetionFlg(false);
            }
        }).catch((exception: AxiosError) => {
            props.onException(exception);
            if(mounted.current) {
                setIconAnimetionFlg(false);
            }
        });
    }

    return (
        <button className="ui blue left floated button" onClick={onClickFunction}>
            <i className={"ui play icon " + (iconAnimetionFlg ? "loading" : "")}></i> 
            {props.children}
        </button>
    )
}

export const FileUploadButton = (props: FileUploadButtonProps) => {
    const [ iconAnimetionFlg, setIconAnimetionFlg ] = useState(false);
  
    function onChangeFunction(e: React.ChangeEvent<HTMLInputElement>) {
        if (e.currentTarget.files === null) {
            return;
        }

        setIconAnimetionFlg(true);
        const file = e.currentTarget.files[0];
        const params = new FormData();
        params.append('file', file);
        axios.post(props.url, params, {
            headers: {
                'content-type': 'multipart/form-data',
            },
        }).then((response: AxiosResponse) => {
            props.onSuccess(response);
            setIconAnimetionFlg(false);
        }).catch((exception: AxiosError) => {
            props.onException(exception);
            setIconAnimetionFlg(false);
        })
    }
  
    return (
        <span>
            <input type="file" className="FileUploadButton" id="FileUploadButton" name="FileUploadButton" onChange={onChangeFunction} />
            <label htmlFor="FileUploadButton" className="ui blue left floated button">
                <i className={"ui upload icon " + (iconAnimetionFlg ? "loading" : "")}></i> 
                { props.children }
            </label>
        </span>
    )
}

export const FileDownloadButton = (props: FileDownloadButtonProps) => {
    const [ iconAnimetionFlg, setIconAnimetionFlg ] = useState(false);
  
    function onClickFunction(e: React.MouseEvent<HTMLButtonElement, MouseEvent>) {
        setIconAnimetionFlg(true);
        const form = new FormData();
        Object.keys(props.params).forEach((key) => form.append(key, props.params[key]));
        axios.post(props.url, form, {
            responseType: "blob",
        }).then((response: AxiosResponse) => {
            const mineType = response.headers["content-type"];
            const blob = new Blob([response.data], { type: mineType });
            const fileURL = window.URL.createObjectURL(blob);
            const fileLink = document.createElement('a');
            fileLink.href = fileURL;
            fileLink.setAttribute('download', props.filename);
            document.body.appendChild(fileLink);
            fileLink.click();

            props.onSuccess(response);
            setIconAnimetionFlg(false);
        }).catch((exception: AxiosError) => {
            props.onException(exception);
            setIconAnimetionFlg(false);
        });
    }
  
    return (
        <button className="ui blue left floated button" onClick={onClickFunction}>
            <i className={"ui download icon " + (iconAnimetionFlg ? "loading" : "")}></i> 
            {props.children}
        </button>
    )
}