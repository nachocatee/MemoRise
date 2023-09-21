// 라이브러리
import axios from "axios";
import { useState } from "react";
import { View, Text, Image, TextInput, Pressable } from "react-native";
import LinearGradient from "react-native-linear-gradient";
import { useDispatch, useSelector } from "react-redux";
import { launchImageLibrary } from "react-native-image-picker";

// 컴포넌트
import GoBackHeader from "../../components/Header/GoBackHeader";
import ProfilePic from "../../components/ProfilePic";
import ConfirmBtn from "../../components/Button/ConfirmBtn";

// 리덕스
import { RootState } from "../../store/store";

// 스타일
import { styles } from "./UserInputStyle";

import { setNickname, setProfileImg } from "../../store/user";

// 백엔드 URL
import { BACKEND_URL, S3_URL } from "../../util/http";
import { StackNavigationProp } from "@react-navigation/stack";
import { RootStackParamList } from "../../App";

// 타입
type ModifyInfoScreenProp = StackNavigationProp<
  RootStackParamList,
  "ModifyInfo"
>;

type Props = {
  navigation: ModifyInfoScreenProp;
};

const ModifyInfoScreen: React.FC<Props> = ({ navigation }) => {
  const dispatch = useDispatch();

  // 리덕스에 저장된 사용자 정보 가져오기
  const tempNickname = useSelector(
    (state: RootState) => state.userInfo.nickname
  );
  const tempProfileImg = useSelector(
    (state: RootState) => state.userInfo.profile_img
  );

  // 닉네임 & 프로필사진 상태관리 (리덕스에 닉네임 & 프로필사진 있다면 초기값으로 사용)
  const [userNickname, setUserNickname] = useState(tempNickname);
  const [userProfileImg, setUserProfileImg] = useState(tempProfileImg);

  // 사용자 프로필 사진 가져오기 (사용자 앨범 접근)
  const selectProfileImageHanlder = () => {
    launchImageLibrary(
      {
        mediaType: "photo",
        maxWidth: 512,
        maxHeight: 512,
        includeBase64: true,
      },
      (response: any) => {
        if (response.didCancel) {
          return;
        } else if (response.errorCode) {
          console.log("Image Error : " + response.errorCode);
        }

        // 백엔드 연동을 위한 Form Data
        const formData = new FormData();
        formData.append("files", {
          uri: response.assets[0].uri,
          type: response.assets[0].type,
          name: response.assets[0].fileName,
        });

        // S3에 사진 업로드
        axios
          .post(BACKEND_URL + "/user/upload", formData, {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          })
          .then((response: any) => {
            // 요청 성공 시, 리덕스 및 상태관리 (사용자 이미지 S3링크로 저장)
            const tempS3URL = S3_URL + response.data[0].savedFileName;
            console.log(tempS3URL);
            setUserProfileImg(tempS3URL);
            dispatch(setProfileImg(tempS3URL));
          })
          .catch((error) => {
            console.log(error);
          });
      }
    );
  };

  // 정보 수정 완료 함수
  const ConfirmHandler = async () => {
    // 사용자 닉네임 리덕스 저장
    dispatch(setNickname(userNickname));

    //백엔드 연동에 필요한 데이터
    const userData = {
      nickname: userNickname,
      profile: userProfileImg,
    };

    // 백엔드 연동
    await axios
      .put(BACKEND_URL + "/user", userData)
      .then((response) => {
        console.log(response);
        if (response.data.success === true) {
          navigation.navigate("Menu");
        } else {
          console.log("회원정보 수정 중 에러가 발생했습니다.");
          navigation.navigate("NotFound");
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
      <GoBackHeader />
      <View style={styles.container}>
        <View>
          <View style={{ alignItems: "center" }}>
            <View style={styles.profilebox}>
              <ProfilePic />
              {userProfileImg ? (
                <Image
                  source={{ uri: userProfileImg }}
                  style={styles.userImage}
                />
              ) : null}
              <Pressable onPress={() => selectProfileImageHanlder()}>
                <Image
                  source={require("../../assets/icons/album.png")}
                  style={styles.album}
                />
              </Pressable>
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
          <ConfirmBtn onPress={ConfirmHandler}>확인</ConfirmBtn>
        </View>
      </View>
    </LinearGradient>
  );
};

export default ModifyInfoScreen;
