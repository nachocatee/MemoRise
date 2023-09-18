import { View, Image, Pressable } from "react-native";
import LinearGradient from "react-native-linear-gradient";
import KakaoLogins, {
  login,
  getProfile,
} from "@react-native-seoul/kakao-login";
import axios from "axios";

import { styles } from "./LoginStyle";

const LoginScreen = () => {
  // 카카오 로그인 함수
  const loginHandler = async () => {
    return await login()
      .then((response) => {
        console.log(response);
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
          <Pressable style={styles.kakao} onPress={loginHandler}>
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
