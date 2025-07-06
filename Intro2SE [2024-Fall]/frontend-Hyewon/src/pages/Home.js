import Header from "../components/Header";
import styles from "../styles/Home.module.css";

import { FaCircleCheck } from "react-icons/fa6";
import { FaChevronLeft } from "react-icons/fa6";
import { FaChevronRight } from "react-icons/fa6";
import { IoTriangleSharp } from "react-icons/io5";
import { FaPlus } from "react-icons/fa6";

import qrCode from "../assets/images/qr_code.png";
import scheduleIcon from "../assets/images/schedule_icon.png";
import scrapIcon from "../assets/images/scrap_icon.png";

import {useState, useEffect} from "react";
import { useLocation, useNavigate } from "react-router-dom";


function Home({student_data}){
    const accessToken = localStorage.getItem('access_token');
    console.log(accessToken); 
  const [count, setCount] = useState(180);
  const navigate = useNavigate();
  const location = useLocation();

  const access_token = location.state.access_token ? location.state.access_token : "";

  useEffect(() => {
    const id = setInterval(() => {
      setCount((count) => count - 1);
    }, 1000);
    
    if(count === 0) {
        clearInterval(id);
        setCount(180);
    }
    return () => clearInterval(id);
  }, [count]);


    return(
       <div className="relative w-full h-full bg-white">
            <Header page="Home" access_token={access_token}/>
            
            <div className="flex justify-between space-x-1 text-center home_tabs bg-white">
                <button className="w-full pt-1.5 pb-2 text-sm font-semibold text-white bg-blue-900 id_tab cursor-pointer">신분증</button>
                <button className="w-full pt-1.5 pb-2 text-sm font-semibold text-white bg-zinc-300 kingo_tab cursor-pointer">KINGO ⓘ</button>
                <button className="w-full pt-1.5 pb-2 text-sm font-semibold text-white bg-zinc-300 favor_tab cursor-pointer">즐겨찾기</button>
            </div>

            <div className="id_card p-4 pt-4.5 m-3 mb-5 bg-neutral-200 rounded-lg shadow-lg">
                <div className={`${styles.borderBottom} flex pb-2 border-neutral-400 justify-between space-x-4 id_info bg-inherit`}>
                    <div className="flex flex-col flex-none std_pic bg-inherit">
                        <div className={`${styles.text} w-10 ml-0 mb-1 text-white font-semibold bg-blue-900 rounded-xl`}>학생증</div>
                        <img className="rounded-md" src="" alt="std_pic" width="90px" height="90px" />
                    </div>
                    <div className="flex flex-col flex-auto std_info bg-inherit">
                        <div className={`flex-none ${styles.text2} w-9 ml-0 mt-0.5 text-blue-950 font-semibold bg-slate-50 rounded-sm`}>
                            신분증 안내
                        </div>
                        <div className="flex-auto flex flex-col text-xs justify-around ml-0 bg-neutral-200">
                            <div className="bg-neutral-200 text-left font-bold m-0">{"홍길동"}</div>
                            <div className="bg-neutral-200">
                                <p className={`${styles.text3} bg-neutral-200 text-left `}>{"소프트웨어학과"}</p>
                                <p className={`${styles.text4} bg-neutral-200 text-left `}>{"0000000000"}</p>
                            </div>
                        </div>
                    </div>
                    <div className="flex-none bg-neutral-200">
                        <FaCircleCheck className="float-left mt-0.5 h-3 bg-neutral-200 text-green-400 "/>
                        <p className={`${styles.text3} font-bold pt-0.5 bg-neutral-200 float-left`}>주 신분증</p>
                    </div>
                </div>
                <div className="id_qr bg-inherit pt-4">
                    <div className="flex bg-inherit">
                        <FaChevronLeft className="bg-inherit text-3xl font-thin my-auto text-gray-400 cursor-pointer"/>
                        <img src={qrCode} alt="qr_code" width="150px" height="150px"/>
                        <FaChevronRight className="bg-inherit text-3xl font-thin my-auto text-gray-400 cursor-pointer"/>
                    </div>
                    <div className="mt-1 bg-inherit">
                        <p className={`inline ${styles.text5} bg-inherit font-semibold`}>남은시간 : </p>
                        <p className={`inline text-xs text-lime-500 bg-inherit font-semibold`}>
                            {Math.floor(count/60)}:{count % 60 < 10 ? '0'+count%60 : count%60}
                        </p>
                    </div>
                </div>
            </div>

            <div className="flex justify-between bg-inherit mx-5">
                <div className="flex-none m-0 flex justify-start space-x-1 bg-inherit">
                    <div className={`${styles.text6} px-3 bg-lime-500 text-white font-semibold rounded`}>NFC</div>
                    <div className={`${styles.text7} px-2 text-blue-950 font-semibold rounded`}>SECOM</div>
                </div>
                <div className="flex-auto bg-inherit">
                    <div className={`${styles.text8} w-fit px-3 rounded-xl`}>
                        <p className="inline bg-inherit text-blue-800 font-semibold">01</p>
                        <p className="inline bg-inherit text-gray-400 font-semibold">/01</p>
                    </div>
                </div>
                <div className={`flex-none w-fit mr-0 ${styles.text7} px-2 text-blue-950 font-semibold rounded`}>SECOM GUIDE</div>
            </div>

            <div className="absolute bottom-16 w-full bg-transparent">
                <div className="flex justify-start space-x-3 h-16 px-4 mx-3 mt-10 box-border bg-neutral-200 rounded-lg shadow-lg">
                    <div className={`flex-none ${styles.shortcut} w-fit p-2 ml-0 my-auto rounded-full bg-transparent cursor-pointer`}>
                        <img className="bg-transparent" src={scheduleIcon} width="25px" height="25px" onClick={() => { console.log("access_token", access_token); navigate("/schedule", { state: { access_token } }); }}/>
                    </div>
                    <div className={`flex-none ${styles.shortcut} w-fit px-2 py-1.5 ml-0 my-auto rounded-full bg-transparent cursor-pointer`}>
                        <img className="bg-transparent" src={scrapIcon} width="25px" height="25px" onClick={()=>{console.log("access_token", access_token);navigate("/keywordRegister", {state:{access_token: access_token}})}}/>
                    </div>
                    <div className={`flex-none ${styles.shortcut} w-fit p-2 ml-0 my-auto rounded-full bg-transparent cursor-pointer`}>
                        <FaPlus className="bg-transparent text-gray-400 my-0.5"/>
                    </div>
                    <div className="flex-auto bg-inherit">
                    </div>
                </div>
            </div>

            <IoTriangleSharp className="fixed inset-x-1/2 bottom-8 bg-transparent text-lime-500"/>
            <div className="fixed bottom-0 w-full h-9 pt-1.5 bg-lime-500 text-white font-bold cursor-pointer" style={{maxWidth:"400px"}}>
                WooriBank / SAMSUNG Wallet
            </div>

       </div> 
    )

}

export default Home;