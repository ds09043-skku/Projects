import React, { useState, useEffect } from "react";
import "../styles/Scrap.css";
import Header from "../components/Header";
import { IoBookmarkSharp } from "react-icons/io5";
import sort from "../assets/images/sort.png";
import kingoMIcon from "../assets/images/kingo-M.png";
import { useLocation, useNavigate } from "react-router-dom";
import axios from "axios";

function ScrapNotifications() {
  const navigate = useNavigate();
  const location = useLocation();

  const access_token = localStorage.getItem('access_token');
  console.log(access_token); 
  const access_token_with_header = "Bearer " + access_token;

  const [currentView, setCurrentView] = useState("scrap");
  const [message, setMessage] = useState("");

  const [scrappedNotifications, setScrappedNotifications] = useState([]);
  //   {
  //     id: 1,
  //     title: "2024학년도 2학기 수강신청 공지사항",
  //     date: "2024-02-15",
  //     isBookMarked: true,
  //     url:"https://www.skku.edu/skku/campus/skk_comm/notice01.do?mode=view&articleNo=122612&article.offset=0&articleLimit=10&srSearchVal=2024%ED%95%99%EB%85%84%EB%8F%84+%ED%95%99%EC%82%AC%EA%B3%BC%EC%A0%95+%EA%B2%A8%EC%9A%B8+%EA%B3%84%EC%A0%88%EC%88%98%EC%97%85+%EC%9A%B4%EC%98%81+%EC%95%88%EB%82%B4"
  //   },
  //   {
  //     id: 2,
  //     title: "2024학년도 2학기 수강철회 안내",
  //     date: "2024-02-14",
  //     isBookMarked: true,
  //     url:""
  //   },
  // ]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        console.log("Fetching scrapped notices...");
        const response = await axios.get(`http://127.0.0.1:5000/user/scrap`, {
          headers: { Authorization: access_token_with_header }
        });

        if (response.data.msg === "get scrap notice success") {
          const mappedData = response.data.data.map((item) => ({
            id: item.noti_id,
            title: item.title,
            url: item.url,
            isBookMarked: true,
          }));
          setScrappedNotifications(mappedData);
          setMessage(response.data.msg);
        } else {
          setMessage(response.data.msg);
        }
      } catch (error) {
        if (error.response) {
          setMessage(error.response.data.msg);
        } else {
          setMessage("An error occurred while connecting to the server.");
        }
      }
    };

    fetchData(); // 비동기 함수 호출
  }, []); // 빈 배열: 컴포넌트 마운트 시 한 번만 실행

  const handleBookmarkClick = async (id, e) => {
    setScrappedNotifications((prev) =>
      prev.filter((notif) => notif.id !== id)
    );

    e.preventDefault();

    try {
      const response = await axios.delete(`http://127.0.0.1:5000/user/noti/${id}`,
       { headers: { Authorization: access_token_with_header }}
      );
      console.log("response.data", response.data);
        // 서버로부터 받은 응답 처리
      if (response.data.msg === "delete success") {
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
  };

  const [content, setContent] = useState("");

  // 공지 검색 처리
  const filteredNotices = scrappedNotifications.filter((notice) =>
    notice.title.includes(content) // 제목에 검색어 포함 여부 확인
  );

  const [isAscending, setIsAscending] = useState(true); // 오름차순/내림차순 상태

  const handleSort = () => {
    const sorted = [...scrappedNotifications].sort((a, b) => {
      if (isAscending) return b.title.localeCompare(a.title); // 내림차순
        return a.title.localeCompare(b.title); // 오름차순
        })
    
    setScrappedNotifications(sorted);
    setIsAscending(!isAscending); // 정렬 상태 토글
  };
  

  return (
    <div className="notification-app">
      <Header page="Scrap" />

      {/* Main Content */}
      <div className="main-content">
      <aside className="flex-none sidebar">
          <div className="kingo-logo">
            <img src={kingoMIcon} width={"50px"} height={"75px"} alt="KINGO-M"/>
            <div className="iconTitle font-bold text-l relative bottom-3">KINGO-M</div>
          </div>
          <ul>
            <li 
                className={`menu-item ${currentView === 'main' ? 'active' : ''}`}
                onClick={() => {setCurrentView('main'); navigate("/scrapSchedule", {state:{access_token: access_token}})}}
            >
                알림 공지
            </li>
            <li 
                className={`menu-item ${currentView === 'scrap' ? 'active' : ''}`}
                onClick={() => {setCurrentView('scrap'); navigate("/scrapNotice", {state:{access_token: access_token}})}}
            >
                스크랩 공지
            </li>
            {/* <li 
                className={`menu-item ${currentView === 'trash' ? 'active' : ''}`}
                onClick={() => setCurrentView('trash')}
            >
                휴지통
            </li> */}
          </ul>
        </aside>

        <div className="notifications">
          <div className="notifications-header bolder">
            <h2>스크랩 공지</h2>
          </div>

          <div className="search-bar">
            <input
              type="text"
              className="search-input"
              placeholder="검색어를 입력하세요"
              value={content} onChange={(e)=>setContent(e.target.value)}
            />
            <button className="search-button">검색</button>
          </div>

          <div className="float-start mb-2 sub_func w-full">
            <div className=" text-xs  pt-1 cursor-pointer" onClick={handleSort}>
                <span className="inline-block float-left" >정렬</span>
                <img className="inline-block float-left" src={sort} width="17px" height="17px" />
            </div>
          </div>


          <ul className="scrap-notification-list">
            {(filteredNotices ? filteredNotices : scrappedNotifications).map((notif) => (
              <li key={notif.id} className="scrap-notification-item relative cursor-pointer ">
                {/*<div className="flex items-center justify-between">*/}
                <div className="scrap-notificaton-title bg-inherit"
                  onClick={()=>navigate("/srapSchedule/relatedNotice/detail", {state:{title: notif.title, noticeURL: notif.url, page:"scrapScheduleNotice", access_token: access_token}})}
                >{notif.title}</div>
                    <div
                        className={`bg-inherit notification-bookmark ${
                            notif.isBookmarked ? "gray" : ""
                          }`}
                          onClick={(e) => handleBookmarkClick(notif.id, e)}
                        >
                    <IoBookmarkSharp className="bg-inherit"/>
                    </div>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default ScrapNotifications;
