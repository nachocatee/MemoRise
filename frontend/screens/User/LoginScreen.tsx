import { View, Image, Pressable } from "react-native";
import LinearGradient from "react-native-linear-gradient";
import KakaoLogins, {
  login,
  getProfile,
} from "@react-native-seoul/kakao-login";
import axios from "axios";

import { styles } from "./LoginStyle";

const LoginScreen = () => {
  // 카카오 사용자 정보 조회 함수
  const getProfileHandler = () => {
    getProfile()
      .then((response) => {
        console.log(response.email);
        // 백엔드에 이메일 정보 보내서 사용자 확인
        axios
          .post("http://j9b106.p.ssafy.io:8000/auth/oauth2/code/kakao", {
            email: response.email,
          })
          .then((response) => {
            console.log(response);
          })
          .catch((error) => {
            console.log(error);
          });
      })
      .catch((error) => {
        console.log(error);
      });
  };

  // 카카오 소셜 로그인 함수
  const loginHandler = async () => {
    await login()
      .then((response) => {
        console.log(response.scopes);
        // 사용자 프로필 정보 조회
        getProfileHandler();
      })
      .catch((error) => {
        console.log(error);
      });
  };

  return (
    <LinearGradient
      //   colors={["#F5F5F5", "red"]}
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <View style={styles.rootContainer}>
        <View style={{ alignItems: "center" }}>
          <Image
            source={require("../../assets/image/note_sm.png")}
            style={styles.image}
          />
          <Image
            source={require("../../assets/image/logo/logo.png")}
            style={styles.logo}
          />
          <Pressable style={styles.kakao} onPress={() => loginHandler()}>
            <Image
              source={require("../../assets/image/kakao.png")}
              style={styles.kakaoImage}
            />
          </Pressable>
        </View>
      </View>
    </LinearGradient>
  );
};

export default LoginScreen;
