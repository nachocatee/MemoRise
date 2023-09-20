import axios from "axios";

// 백엔드 URL 주소
export const BACKEND_URL = "http://j9b106.p.ssafy.io:8000";

// 로그인 요청 AXIOS
export const loginUserCheckHandler = async (email: string) => {
  const response = await axios.post(BACKEND_URL + "/auth", email);

  return response.data;
};
