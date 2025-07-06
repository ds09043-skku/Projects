import { IoIosArrowBack, IoIosGlobe } from "react-icons/io";
import { AiOutlineSend } from "react-icons/ai"; 
import { useState } from "react";
import skku_logo2 from "../assets/images/skku_logo2.png";
import chatbotIcon from "../assets/images/chatbot_icon.png";
import Header from "../components/Header";
import axios from "axios";
import { useLocation } from "react-router-dom";

function Chatbot({ onBack }) {

  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([
    { text: "안녕하세요? 저는 킹고봇입니다. 무엇을 도와드릴까요?", sender: "bot" },
  ]);
  const [input, setInput] = useState("");

  const location = useLocation();
  const access_token = localStorage.getItem('access_token');
console.log(access_token); 
  const access_token_with_header = "Bearer " + access_token;

  const handleSendMessage = async(e) => {
    e.preventDefault(); // 폼 제출 시 페이지 새로고침 방지
    console.log("input", input);

    const input_edited = input.trim();
    try {
      const response = await axios.post(`http://127.0.0.1:5000/chat`,
        {word : input_edited}, 
        {headers: { Authorization: access_token_with_header }
      });
      console.log("response", response);
        // 서버로부터 받은 응답 처리
      if (response.data.msg === "chat response success") {
        console.log("response data", response.data);
        setMessage(response.data.msg); // "Login successful"

        setMessages([...messages, { text: input_edited, sender: "user" }]);
        setInput("");
        setTimeout(() => {
          setMessages((prev) => [...prev, { text: response.data.response, sender: "bot" }]);
        }, 500);
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

    setInput("");

  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* Header Section */}
      <Header page="chatbot"/>

      {/* Bot Info Section */}
      <div className="flex items-center justify-center w-full bg-white text-black p-3 border  border-gray-400 rounded-lg">
        <div className="flex items-center bg-white">
          <img src={chatbotIcon} alt="Chatbot Icon" className="h-6 mr-2" />
          <p className="font-bold text-sm bg-white">KINGOBOT(킹고봇)</p>
        </div>
      </div>

      {/* Messages Section */}
      <div className="flex-grow p-3 overflow-y-auto">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`mb-2 p-3 shadow-md rounded-lg text-sm break-words break-all overflow-auto whitespace-pre-line ${
              message.sender === "user"
                ? "ml-auto mr-2 bg-white text-right max-w-[60%]" // Reduced max width and added `mr-2`
                : "mr-auto bg-white text-left max-w-[70%] ml-2" // Added margin for bot messages
            }`}
          >
            {message.text}
          </div>
        ))}
      </div>

      {/* Input Section */}
      <div className="flex p-3 border-t bg-white fixed bottom-1 w-full" style={{maxWidth:"400px"}}>
        <input
          className="flex-grow p-2 border border-gray-400 rounded-lg text-sm"
          type="text"
          placeholder="메시지를 입력하세요..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <AiOutlineSend
          size={38}
          className="ml-1 text-gray-500 cursor-pointer bg-white"
          onClick={(e)=>handleSendMessage(e)}
        />
      </div>
    </div>
  );
}

export default Chatbot;
