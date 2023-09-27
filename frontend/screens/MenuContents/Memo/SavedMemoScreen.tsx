// 라이브러리
import axios from "axios";
import React, { useEffect } from "react";
import { View, Text, Image } from "react-native";
import { useSelector } from "react-redux";
import LinearGradient from "react-native-linear-gradient";

// 컴포넌트
import GoBackHeader from "../../../components/Header/GoBackHeader";
import { RootState } from "../../../store/store";

// 스타일
import { styles } from "./MemoStyle";

// 통신
import { BACKEND_URL } from "../../../util/http";

const SavedMemoScreen = () => {
  // 타입
  type BookMarkMemoProps = {
    accessType: string;
    content: string;
    file: string | null;
    isBookmarked: boolean;
    itemImage: string | null;
    memoSeq: number;
    nickname: string;
    updateAt: string;
  };

  // 유저 ID
  const userId = useSelector((state: RootState) => state.userInfo.id);

  // 백엔드 통신
  useEffect(() => {
    console.log(userId);
    const fetchData = async () => {
      await axios
        // .get(BACKEND_URL + `/user/${userId}/bookmarks`)
        // 쫀디기
        .get(BACKEND_URL + "/user/23/bookmarks")
        .then((response) => {
          console.log(response.data);
        })
        .catch((error) => {
          console.log(error);
        });
    };
    fetchData();
  }, []);

  return (
    <LinearGradient
      // colors={["#F5F5F5", "red"]}
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <GoBackHeader />
      <View>
        <Text style={styles.title}>저장된 메모</Text>
        <View style={styles.container}>
          <Image
            style={styles.icon}
            source={require("../../../assets/icons/memo_empty.png")}
          />
          <Text style={styles.empty}>저장된 메모가 없습니다</Text>
        </View>
      </View>
    </LinearGradient>
  );
};

export default SavedMemoScreen;
