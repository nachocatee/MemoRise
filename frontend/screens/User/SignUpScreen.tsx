// 라이브러리
import axios from "axios";
import React, { useState } from "react";
import { View, Text, Image, TextInput } from "react-native";
import LinearGradient from "react-native-linear-gradient";
import { StackNavigationProp } from "@react-navigation/stack";
import { useDispatch, useSelector } from "react-redux";

// 컴포넌트
import ConfirmBtn from "../../components/Button/ConfirmBtn";
import HighlightHeader from "../../components/Header/HighlightHeader";
import ProfilePic from "../../components/ProfilePic";
import { RootStackParamList } from "../../App";
import { styles } from "./UserInputStyle";

// 리듀서
import { setNickname, setProfileImg } from "../../store/user";

// 백엔드 URL
import { BACKEND_URL } from "../../util/http";

type SignUpScreenNavigationProp = StackNavigationProp<
  RootStackParamList,
  "SignUp"
>;

type Props = {
  navigation: SignUpScreenNavigationProp;
};

const SignUpScreen: React.FC<Props> = ({ navigation }) => {
  const dispatch = useDispatch();

  // 리덕스에 저장된 사용자 정보 가져오기
  const userEmail = useSelector((state: any) => state.userInfo.email);
  const tempNickname = useSelector((state: any) => state.userInfo.nickname);
  const tempProfileImg = useSelector(
    (state: any) => state.userInfo.profile_img
  );

  // 닉네임 & 프로필사진 상태관리 (리덕스에 닉네임 & 프로필사진 있다면 초기값으로 사용)
  const [userNickname, setUserNickname] = useState(tempNickname);
  const [userProfileImg, setUserProfileImg] = useState(tempProfileImg);

  const SignUpHandler = async () => {
    //리덕스 정보 저장
    dispatch(setNickname(userNickname));

    console.log(userEmail, userNickname, userProfileImg);

    // 백엔드 연동
    const userData = {
      email: userEmail,
      nickname: userNickname,
      profile: userProfileImg,
    };

    await axios
      .post(BACKEND_URL + "/user", userData)
      .then((response) => {
        console.log(response.data);
        //메인페이지 이동
        if (response.data.success === true) {
          navigation.navigate("Main");
        } else {
          console.log("회원가입 중 에러가 발생했습니다.");
        }
      })
      .catch((error) => {
        console.log(error);
      });
  };

  return (
    <LinearGradient
      // colors={["#F5F5F5", "red"]}
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <HighlightHeader />
      <View style={styles.container}>
        <View>
          <View style={{ alignItems: "center" }}>
            <View style={styles.profilebox}>
              <ProfilePic />
              <Image
                source={require("../../assets/icons/album.png")}
                style={styles.album}
              />
            </View>
          </View>
          <View style={styles.inputBox}>
            <Text style={styles.text}>닉네임</Text>
            <TextInput
              style={styles.input}
              value={userNickname}
              onChangeText={setUserNickname}
            />
            <Text style={styles.infoText}>
              한글, 영어, 숫자만 사용할 수 있어요. (최대 10자)
            </Text>
          </View>
        </View>

        <View style={styles.buttonContainer}>
          <ConfirmBtn onPress={SignUpHandler}>회원가입</ConfirmBtn>
        </View>
      </View>
    </LinearGradient>
  );
};

export default SignUpScreen;
