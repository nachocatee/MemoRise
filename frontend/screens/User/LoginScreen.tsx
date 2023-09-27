//라이브러리
import React from "react";
import axios from "axios";
import { useDispatch } from "react-redux";
import { useNavigation } from "@react-navigation/native";
import { StackNavigationProp } from "@react-navigation/stack";
import { View, Image, Pressable } from "react-native";
import KakaoLogins, {
  login,
  getProfile,
} from "@react-native-seoul/kakao-login";
import LinearGradient from "react-native-linear-gradient";
import AsyncStorage from "@react-native-async-storage/async-storage";

// 스타일
import { styles } from "./LoginStyle";

//리듀서
import {
  setEmail,
  setNickname,
  setProfileImg,
  setUserId,
} from "../../store/user";
import { BACKEND_URL } from "../../util/http";

type RootStackParamList = {
  SignUp: undefined;
  Main: undefined;
};

const LoginScreen = () => {
  // 네비게이션
  const navigation = useNavigation<StackNavigationProp<RootStackParamList>>();

  const dispatch = useDispatch();

  // 카카오 사용자 정보 조회 함수
  const getProfileHandler = async () => {
    const response = await getProfile();

    // 닉네임 & 프로필사진 여부 판단 후 리덕스 저장
    if (response.nickname !== null) {
      dispatch(setNickname(response.nickname));
    }
    if (response.profileImageUrl !== null) {
      dispatch(setProfileImg(response.profileImageUrl));
    }

    // 백엔드에 이메일 정보 보내서 사용자 확인
    await axios
      .post(BACKEND_URL + "/auth", {
        email: response.email,
      })
      .then((response) => {
        const checkUser = response.data;
        // 기존 사용자 여부에 따라 네비게이션 이동
        if (checkUser.success === true) {
          // 유저 아이디 리덕스에 저장
          dispatch(setUserId(checkUser.userSeq));
          dispatch(setNickname(checkUser.nickName));
          dispatch(setProfileImg(checkUser.profile));
          navigation.navigate("Main");
        } else {
          navigation.navigate("SignUp");
        }
      })
      .catch((error) => {
        console.log(error);
      });

    // 리덕스 : 사용자 이메일값 변경
    dispatch(setEmail(response.email));
  };

  // 카카오 소셜 로그인 함수
  const loginHandler = async () => {
    await login()
      .then((response) => {
        // 사용자 토큰
        console.log(response);
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
