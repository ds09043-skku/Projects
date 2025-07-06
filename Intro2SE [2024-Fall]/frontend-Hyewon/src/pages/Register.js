import React, { useEffect, useState } from "react";
import { IoIosArrowBack } from "react-icons/io";
import { IoBookmarkSharp } from "react-icons/io5";
import ExternalPage from "../components/ExternalPage";
import Header from "../components/Header";
import { BiBorderBottom } from "react-icons/bi";
import { useLocation, useNavigate } from "react-router-dom";
import axios from "axios";


function Register({ onBack }) {
  const navigate = useNavigate();
  const location = useLocation();
  
  const access_token = localStorage.getItem('access_token');
  console.log(access_token); 

  const [itemColors, setItemColors] = useState(["gray", "gray", "gray", "gray"]);
  const access_token_with_header = "Bearer " + access_token;

  const [keywords, setKeywords] = useState([]);
  const [message, setMessage] = useState("");
  const [scrap, setScrap] = useState({});

  useEffect(()=>{
    const fetchData = async () =>{
      // e.preventDefault(); // 폼 제출 시 페이지 새로고침 방지
      console.log("inputKeyword", inputKeyword);

      try {
        const response = await axios.get("http://127.0.0.1:5000/user/keyword",
          { headers: { Authorization: access_token_with_header } }
        );
        console.log("response", response);
          // 서버로부터 받은 응답 처리
          if (response.data.msg === "get registerd keyword success") {
            console.log("response data", response.data);
            setKeywords(response.data.data);
  
            // const updatedItemColors = response.data.data.map(data =>
            //   data.scrap === true ? "red" : "grey"
            // );
    
            // console.log("updatedItemColors", updatedItemColors)
            // setItemColors(updatedItemColors);
            setMessage(response.data.msg); // "register keyword successful"
          } else {
            setMessage(response.data.msg); // "Invalid credentials"
          }
      } catch (error) {
        // 에러 처리
        if (error.response) {
          setMessage(error.response.data.msg); // 서버에서 보낸 에러 메시지
        } else {
          setMessage("An error occurred while connecting to the server.");
        }
      }
    }
    fetchData();
  }, [])
  
  const [inputKeyword, setInputKeyword] = useState("");
  const [isAlertEnabled, setIsAlertEnabled] = useState(false);
  const [filteredNotices, setFilteredNotices] = useState([]);
  const [clickedKeyword, setClickedKeyword] = useState("");
  const [selectedNoticeURL, setSelectedNoticeURL] = useState(null); // New state for selected notice URL

  const notices = [
    {
      title: "2024학년도 학사과정 겨울 계절수업 운영 안내",
      url: "https://www.skku.edu/skku/campus/skk_comm/notice01.do?mode=view&articleNo=122612&article.offset=0&articleLimit=10&srSearchVal=2024%ED%95%99%EB%85%84%EB%8F%84+%ED%95%99%EC%82%AC%EA%B3%BC%EC%A0%95+%EA%B2%A8%EC%9A%B8+%EA%B3%84%EC%A0%88%EC%88%98%EC%97%85+%EC%9A%B4%EC%98%81+%EC%95%88%EB%82%B4",
      scrap:0,
    },
    {
      title: "[반도체소부장혁신융합대학사업단] 2024학년도 겨울계절학기 수강신청 안내",
      url: "https://www.skku.edu/skku/campus/skk_comm/notice01.do?mode=view&articleNo=122309&article.offset=0&articleLimit=10&srSearchVal=%EB%B0%98%EB%8F%84%EC%B2%B4%EC%86%8C%EB%B6%80%EC%9E%A5%ED%98%81%EC%8B%A0%EC%9C%B5%ED%95%A9%EB%8C%80%ED%95%99%EC%82%AC%EC%97%85%EB%8B%A8%5D+2024%ED%95%99%EB%85%84%EB%8F%84+%EA%B2%A8%EC%9A%B8%EA%B3%84%EC%A0%88%ED%95%99%EA%B8%B0+%EC%88%98%EA%B0%95%EC%8B%A0%EC%B2%AD+%EC%95%88%EB%82%B4",
      scrap:1,
    },
    {
      title: "[행사/세미나] 글로벌 IT전문가와 킹고인의 만남 시즌2 쉰일곱번째 만남 참가 신청(3/28 목)",
      url: "https://www.skku.edu/skku/campus/skk_comm/notice01.do?mode=view&articleNo=116546&article.offset=0&articleLimit=10&srSearchVal=%5B%ED%96%89%EC%82%AC%2F%EC%84%B8%EB%AF%B8%EB%82%98%5D+",
      scrap:0,
    },
  ];

  const addKeyword = async (e) => {

    e.preventDefault(); // 폼 제출 시 페이지 새로고침 방지
    console.log("inputKeyword", inputKeyword);

    const inputkw = inputKeyword.trim();
    console.log("token", access_token_with_header);
    try {
      const response = await axios.post(
      "http://127.0.0.1:5000/user/keyword",
      { keyword: inputkw, is_calendar: 0 },
      { headers: { Authorization: access_token_with_header } }
      );
      console.log("response", response);
      // 서버로부터 받은 응답 처리
      if (response.data.msg === "regist keyword success") {
        console.log("response data", response.data);
        setMessage(response.data.msg); // "register keyword successful"
        if (!keywords.includes(inputkw)) {
          setKeywords([...keywords, {
            "keyword": inputkw,
            "keywordid": response.data.keyword_id,
            "new": 0
          }]);
        }
      } else {
        setMessage(response.data.msg); // "Invalid credentials"
      }
    } catch (error) {
      // 에러 처리
      if (error.response) {
        setMessage(error.response.data.msg); // 서버에서 보낸 에러 메시지
      } else {
        setMessage("An error occurred while connecting to the server.");
      }
    }

    setInputKeyword("");
  };

  const handleKeywordClick = async (keyword, e) => {
    setClickedKeyword(keyword);

    e.preventDefault(); // 폼 제출 시 페이지 새로고침 방지
    console.log("keyword Click");
    try {
      const response = await axios.get(`http://127.0.0.1:5000/user/${keyword.keywordid}`,
        { headers: { Authorization: access_token_with_header } });
      console.log("response", response);
        // 서버로부터 받은 응답 처리
      if (response.data.msg === "get regist keyword notice success") {
        console.log("response data", response.data);
        setMessage(response.data.msg); // "register keyword successful"
        // const matchingNotices = notices.filter((notice) =>
        //   notice.title.includes(keyword)
        // );
        setFilteredNotices(response.data.data);

        const updatedScrap = response.data.data.reduce((acc, d) => {
          if (d.scrap) {
            acc[d.noti_id] = true; // d.noti_id를 키로, true를 값으로 설정
          }
          return acc; // 누적된 딕셔너리 반환
        }, {});
        setScrap(updatedScrap);

        // const updatedItemColors = response.data.data.map(data =>
        //   data.scrap === true ? "red" : "grey"
        // );

        // console.log("updatedItemColors", updatedItemColors)
        // setItemColors(updatedItemColors);
      } else {
        setMessage(response.data.msg); // "Invalid credentials"
      }
    } catch (error) {
      // 에러 처리
      if (error.response) {
        setMessage(error.response.data.msg); // 서버에서 보낸 에러 메시지
      } else {
        setMessage("An error occurred while connecting to the server.");
      }
    }

  };

  const handleNoticeClick = (url) => {
    setSelectedNoticeURL(url); // Set the selected notice URL
  };

  const handleBackToNotices = () => {
    setSelectedNoticeURL(null); // Clear the selected notice URL
  };

  const handleToggle = () => {
    setIsAlertEnabled((prev) => !prev);
  };

  const toggleIconColor = async (index, e, notice) => {
    console.log("index", index);
    console.log("notice", notice);

    // setItemColors((prevColors) =>
    //   prevColors.map((color, i) =>
    //     i === index ? (color === "gray" ? "red" : "gray") : color
    //   )
    // );    

    e.preventDefault();

    setScrap((prevState) => ({
      ...prevState,
      [notice.noti_id]: !prevState[notice.noti_id], // 해당 이벤트의 별표 상태를 반전시킴
    }));

    if(notice.scrap){
      setFilteredNotices((prev)=>
        prev.map((noti) =>
          noti.id === notice.noti_id ? { ...noti, scrap: 0 } : noti
        )
      )

      try {
        const response = await axios.post(`http://127.0.0.1:5000/user/noti/${notice.noti_id}`,
          { scrap: 0 },
          { headers: { Authorization: access_token_with_header } }
        );
      console.log("response.data", response.data);
          // 서버로부터 받은 응답 처리
      if (response.data.msg === "scrap success") {
          console.log("response data", response.data);
          setMessage(response.data.msg); // "register keyword successful"
      } else {
          setMessage(response.data.msg); // "Invalid credentials"
      }
      } catch (error) {
      // 에러 처리
      if (error.response) {
          setMessage(error.response.data.msg); // 서버에서 보낸 에러 메시지
      } else {
          setMessage("An error occurred while connecting to the server.");
      }
      }
  }
  else{
      setFilteredNotices((prev)=>
        prev.map((noti) =>
          noti.id === notice.noti_id ? { ...noti, scrap: 1 } : noti
        )
      )
      try {
          const response = await axios.post(`http://127.0.0.1:5000/user/noti/${notice.noti_id}`,
            { scrap: 1 },
            { headers: { Authorization: access_token_with_header } }
          );
          console.log("response.data", response.data);
              // 서버로부터 받은 응답 처리
          if (response.data.msg === "scrap success") {
              console.log("response data", response.data);
              setMessage(response.data.msg); // "register keyword successful"
          } else {
              setMessage(response.data.msg); // "Invalid credentials"
          }
          } catch (error) {
          // 에러 처리
          if (error.response) {
              setMessage(error.response.data.msg); // 서버에서 보낸 에러 메시지
          } else {
              setMessage("An error occurred while connecting to the server.");
          }
          }
    }
  };

  return (
    <div className="bg-gray-50 h-screen">
     {selectedNoticeURL ? (
  // Render the ExternalPage if a notice is selected
  <div className="h-full flex flex-col">
    {/* Header Section */}
    <Header page="keywordRegister" access_token={access_token} />

    {/* External Page */}
    <ExternalPage url={selectedNoticeURL} />
  </div>
) : (

        <div className="bg-white">
          {/* Header Section */}
          <Header page="keywordRegister" />

          <hr style={{background:"gray"}}/>

          {/* Input Section */}
          <div className="flex items-center mt-5 mx-4 bg-white">
            <input
              type="text"
              value={inputKeyword}
              onChange={(e) => setInputKeyword(e.target.value)}
              placeholder="관심 키워드를 입력하세요."
              className="flex-1 p-2 border border-gray-300 rounded-md shadow-sm"
            />
            <button
              onClick={(e) => addKeyword(e)}
              className="ml-2 px-4 py-2 bg-blue-600 text-white rounded-md shadow-sm"
            >
              등록
            </button>
          </div>

          {/* Notices Section */}
          <div className="flex mt-5 mx-4 border border-gray-300 rounded-lg h-[calc(100vh-180px)] p-4" style={{backgroundColor:"#f2f2f2"}}>
            {/* Left Section */}
            <div className="flex-none flex flex-col pr-4 pt-3" style={{borderRight:"0.5px darkgray solid"}}>
              <h3 className="font-semibold text-center border-b pb-3 w-fit pt-1">
                나의 키워드
              </h3>
              <ul className="pt-3 space-y-2" style={{minHeight:"100px", backgroundColor:"#dee2e6", borderRadius:"5px"}}>
                {keywords.map((keyword, index) => (
                  <li key={keyword.keywordid} className="text-xs bg-inherit">
                    <span
                      className={`cursor-pointer ${
                        clickedKeyword === keyword
                          ? "font-bold text-blue-600"
                          : ""
                      } bg-inherit`}
                      onClick={(e) => handleKeywordClick(keyword, e)}
                    >
                      {keyword.keyword}
                    </span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Right Section */}
            <div className="flex-auto p-4 pr-0">
              <h3 className="font-semibold text-center border-b pb-2" style={{borderBottom:"0.5px darkgray solid"}}>
                관련 공지
              </h3>
              <ul className="mt-4 space-y-2">
                {filteredNotices.length === 0 ? (
                  <p className="text-gray-500 text-center">
                    관련 공지가 없습니다.
                  </p>
                ) : (
                  filteredNotices.map((notice, index) => (
                    <li
                      key={index}
                      className="flex p-2 border rounded-md bg-gray-200 flex justify-between"
                    >
                      <span style={{fontSize:"10px"}}
                        className="flex-auto cursor-pointer bg-inherit pr-1"
                        // onClick={() => handleNoticeClick(notice.url)}
                        onClick={()=>navigate("/keyword/notice", {state:{title: notice.title, noticeURL: notice.url, page:"keywordRegister", access_token: access_token}})}
                      >
                        {notice.title}
                      </span>
                      <div className="flex-none my-auto bg-inherit">
                        <IoBookmarkSharp
                          size={20}
                          className={`bg-inherit cursor-pointer ${
                            scrap[notice.noti_id]
                              ? "text-red-500"
                              : "text-gray-500"
                          }`}
                          style={{width:"15px"}}
                          onClick={(e) => toggleIconColor(index, e, notice)}
                        />
                      </div>
                      
                    </li>
                  ))
                )}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Register;