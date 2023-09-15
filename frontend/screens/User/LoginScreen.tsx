import { View, Image, StyleSheet, Pressable } from "react-native";
import LinearGradient from "react-native-linear-gradient";
import KakaoLogins, {
  login,
  getProfile,
} from "@react-native-seoul/kakao-login";
import axios from "axios";

import { calculateDynamicWidth } from "../../constants/dynamicSize";

const LandingScreen = () => {
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
            style={imageStyle}
          />
          <Image
            source={require("../../assets/image/logo/logo.png")}
            style={[logoStyle, styles.logo]}
          />
          <Pressable style={styles.kakao} onPress={loginHandler}>
            <Image
              source={require("../../assets/image/kakao.png")}
              style={kakaoStyle}
            />
          </Pressable>
        </View>
      </View>
    </LinearGradient>
  );
};

export default LandingScreen;

const styles = StyleSheet.create({
  rootContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  logo: {
    position: "absolute",
    top: "27%",
  },
  kakao: {
    position: "absolute",
    bottom: "27%",
  },
});

const imageStyle = {
  width: calculateDynamicWidth(320),
  height: calculateDynamicWidth(333),
};

const logoStyle = {
  width: calculateDynamicWidth(253),
  height: calculateDynamicWidth(48),
};

const kakaoStyle = {
  width: calculateDynamicWidth(230),
  height: calculateDynamicWidth(48),
};
