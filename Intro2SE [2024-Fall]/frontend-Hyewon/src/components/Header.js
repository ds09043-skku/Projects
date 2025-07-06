import { PiList } from "react-icons/pi";
import { IoIosSearch } from "react-icons/io";
import { IoIosArrowBack } from "react-icons/io";
import skkuLogo from "../assets/images/skku_logo.png";
import skkuLogo2 from "../assets/images/skku_logo2.png";
import chatbotIcon from "../assets/images/chatbot_icon.png";
import noticeIcon from "../assets/images/notice_icon.png";

import { useNavigate, useLocation } from 'react-router-dom';

function Header({page, title, scheduleTitle, notice, selectedFavorites, selectedMark, prevPage, detail, access_token}){
    const navigate = useNavigate();

    console.log(page, title);
    console.log(scheduleTitle, notice);

    if(page == "Home"){
        return(
            <div className="flex flex-unwrap justify-between space-x-2 w-full h-14 p-3 text-3xl border-b-stone-400 bg-white">
                <div className="flex-1 bg-white"><PiList className="m-0 bg-inherit cursor-pointer" /></div>
                <div className="flex-auto flex justify-center space-x-2 bg-white cursor-pointer">
                    <img className="w-fit m-0" src={skkuLogo2} alt="skku_logo" height="25px" width="" />
                    <p className="pt-1 bg-inherit text-base font-bold">성균관대학교</p>
                </div>
                <div className="flex-none flex justify-center space-x-1 bg-white">
                    <img className="w-fit m-0 cursor-pointer" src={chatbotIcon} alt="chatbot_icon" height="25px" width="" onClick={()=>navigate("/chatbot", {state:{access_token: access_token}})} />
                    <img className="w-fit m-0 cursor-pointer" src={noticeIcon} alt="notice_icon" height="25px" width="" onClick={()=>navigate("/scrapSchedule", {state:{access_token: access_token}})}/>
                    <IoIosSearch className="bg-inherit cursor-pointer"/>
                </div>
            </div>
        )
    }
    else if (page == "Schedule"){
        return(
            <div className="flex flex-unwrap justify-center space-x-2 w-full h-14 p-3 text-3xl border-b-stone-400 bg-white">
                <IoIosArrowBack className="flex-none bg-inherit mt-1 cursor-pointer" onClick={() => navigate("/home", {state:{access_token: access_token}})}/>
                <div className="flex flex-auto justify-center gap-1 bg-inherit h-full">
                    <img className="flex-none h-full m-0" src={skkuLogo2} alt="skku_logo"  />
                    <p className="flex-none pt-1 bg-inherit text-base font-bold">2024년 학사일정표</p>
                </div>
            </div>
        )
    }
    else if (page == "ScheduleDetail"){
        return(
            <div className="flex flex-unwrap justify-center space-x-2 w-full h-14 p-3 text-3xl border-b-stone-400 bg-white">
                <IoIosArrowBack className="flex-none bg-inherit mt-1 cursor-pointer" onClick={() => navigate("/schedule", {state:{favorites:selectedFavorites, access_token: access_token}})}/>
                <div className="flex flex-auto justify-center gap-1 bg-inherit h-full">
                    <img className="flex-none h-full m-0" src={skkuLogo2} alt="skku_logo"  />
                    <p className="flex-none pt-1 bg-inherit text-base font-bold">선택 일정 관련 공지사항</p>
                </div>
                
            </div>
        )
    }
    else if(page == "selectedNotice"){
        return(
            <div className="flex flex-unwrap justify-center space-x-2 w-full h-14 p-3 px-4 pl-2 text-3xl border-b-stone-400 bg-white">
                <IoIosArrowBack className="flex-none bg-inherit mt-1 cursor-pointer" onClick={() => navigate(-1, {state:{notice:notice, title:scheduleTitle, selectedMark: selectedMark, access_token: access_token}})}/>
                <div className="flex flex-auto justify-center w-full gap-1 bg-inherit h-full">
                    <img className="flex-none w-fit m-0" src={skkuLogo2} alt="skku_logo" height="25px" width="" />
                    <div className="flex-auto flex flex-row bg-inherit">
                        <div className="flex-auto pt-1 bg-inherit text-base font-bold overflow-hidden bg-inherit">{title}</div>
                        <div className="flex-none pt-1 bg-inherit text-base font-bold overflow-hidden bg-inherit align-middle">...</div>
                    </div>
                </div>
                
            </div>
        )
    }
    else if (page == "Scrap") {
        return (
            <div className="flex flex-wrap justify-between items-center w-full h-14 p-3 text-3xl border-b bg-white">
                {/* 뒤로가기 버튼 */}
                <IoIosArrowBack 
                    className="flex-none bg-inherit cursor-pointer" 
                    onClick={() => navigate(detail=="detail"? -1: "/home", {state:{access_token: access_token}})} 
                />
                <div className="flex flex-auto justify-center gap-1 bg-inherit h-full">
                    <img className="flex-none h-full m-0" src={skkuLogo2} alt="skku_logo"  />
                    <p className="flex-none pt-1 bg-inherit text-base font-bold">나의 공지함</p>
                </div>
                
            </div>
        );
    }
    else if(page == "keywordRegister"){
        return(
        <div className="flex flex-unwrap justify-center space-x-2 w-full h-14 p-3 text-3xl border-b-stone-400 bg-white">
            <IoIosArrowBack className="flex-none bg-inherit mt-1 cursor-pointer" onClick={() => navigate("/home", {state:{access_token: access_token}})}/>
            <div className="flex flex-auto justify-center gap-1 bg-inherit h-full">
                <p className="flex-none pt-1 bg-inherit text-base font-bold">관심 공지 등록 및 확인</p>
            </div>
        </div>)
    }    
    else if (page == "chatbot"){
        return(
        <div className="flex flex-unwrap justify-center space-x-2 w-full h-14 p-3 text-3xl border-b-stone-400 bg-white">
            <IoIosArrowBack className="flex-none bg-inherit mt-1 cursor-pointer" onClick={() => navigate("/home", {state:{access_token: access_token}})}/>
            <div className="flex flex-auto justify-center bg-inherit h-full">
                <img className="flex-none h-full m-0" src={skkuLogo2} alt="skku_logo"  />
                <p className="flex-none pt-1 bg-inherit text-base font-bold">성균관대학교</p>
            </div>
        </div>
        )
    }
    return null;
    
}

export default Header;