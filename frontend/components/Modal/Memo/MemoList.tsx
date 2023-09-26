// 라이브러리
import axios from "axios";
import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  Image,
  Pressable,
} from "react-native";
import LinearGradient from "react-native-linear-gradient";

// 스타일
import { calculateDynamicWidth } from "../../../constants/dynamicSize";
import Colors from "../../../constants/colors";

// 백엔드 통신
import { BACKEND_URL } from "../../../util/http";

// 컴포넌트
import BookMarkBtn from "../../Button/BookMarkBtn";

// 메인페이지 상태관리를 위한 타입 지정
type MemoListProp = {
  onMemoWritePress: () => void;
  onMemoDetailPress: (memoSeq: number) => void;
  id: string;
};

const MemoList: React.FC<MemoListProp> = ({
  onMemoWritePress,
  onMemoDetailPress,
  id,
}) => {
  // FlatList 사용을 위한 Type 지정
  type MemoTypeProps = {
    accessType: string;
    content: string;
    file: string | null;
    isBookmarked: boolean;
    itemImage: string;
    memoSeq: number;
    nickname: string;
    updatedAt: string;
  };

  type MemoListItemProps = {
    item: MemoTypeProps;
  };

  // 북마크 상태관리 및 함수
  const [memoData, setMemoData] = useState<MemoTypeProps[]>([]);

  // 사용자가 작성한 or 태그된 메모 AXIOS
  useEffect(() => {
    axios
      // .get(BACKEND_URL + `/memos/${id}/list/23`)
      .get(BACKEND_URL + `/memos/8ef97a8a0be/list/23`)
      .then((response) => {
        console.log(response);
        setMemoData(response.data);
      })
      .catch((error) => {
        console.log(error);
      });
  }, []);

  // FlatList 사용을 위한 MemoList 정리
  const MemoList: React.FC<MemoListItemProps> = ({ item }) => (
    <View style={styles.memoContainer}>
      <Pressable onPress={() => onMemoDetailPress(item.memoSeq)}>
        <LinearGradient
          colors={["#FFFFFF", "#F5F5F5"]}
          start={{ x: 0, y: 0 }}
          end={{ x: 0, y: 1 }}
          style={{ minHeight: calculateDynamicWidth(104) }}
        >
          <View style={styles.innerContainer}>
            <View style={styles.calenderContainer}>
              <Text style={styles.calendar}>{item.updatedAt}</Text>
            </View>
            {item.file !== null ? (
              <View>
                <Text
                  style={styles.content}
                  numberOfLines={2}
                  ellipsizeMode="tail"
                >
                  {item.content}
                </Text>
                <Image source={{ uri: item.file }} style={styles.photo} />
              </View>
            ) : (
              <Text
                style={styles.content}
                numberOfLines={4}
                ellipsizeMode="tail"
              >
                {item.content}
              </Text>
            )}
            <View style={styles.bottomContainer}>
              <Text style={styles.nickname}>{item.nickname}</Text>
              <Text style={styles.open}>
                {item.accessType === "OPEN"
                  ? "전체공개"
                  : item.accessType === "RESTRICT"
                  ? "일부공개"
                  : "비공개"}
              </Text>
            </View>
          </View>
        </LinearGradient>
      </Pressable>
      <BookMarkBtn
        memoSeq={item.memoSeq}
        detailStyle={[styles.bookmark, styles.bookmarkSize]}
        bookmarkType={item.isBookmarked}
      />
    </View>
  );

  return (
    <>
      <View style={styles.mainContainer}>
        <Pressable style={styles.memoWriteContainer} onPress={onMemoWritePress}>
          <Image
            source={require("../../../assets/icons/memo_write.png")}
            style={styles.memoWrite}
          />
        </Pressable>
        <View style={styles.memoListContainer}>
          <FlatList
            data={memoData}
            renderItem={({ item }) => <MemoList item={item} />}
            keyExtractor={(item) => item.memoSeq.toString()}
          />
        </View>
      </View>
    </>
  );
};

export default MemoList;

const styles = StyleSheet.create({
  mainContainer: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: [
      { translateY: -calculateDynamicWidth(350) / 2 },
      { translateX: -calculateDynamicWidth(306) / 2 },
    ],
  },

  memoWriteContainer: {
    alignItems: "flex-end",
    marginBottom: 3,
  },

  memoWrite: {
    width: calculateDynamicWidth(38),
    height: calculateDynamicWidth(38),
  },

  memoListContainer: {
    height: calculateDynamicWidth(350),
  },

  memoContainer: {
    width: calculateDynamicWidth(306),
    minHeight: calculateDynamicWidth(104),
    maxHeight: calculateDynamicWidth(185),
    borderRadius: calculateDynamicWidth(15),
    marginBottom: calculateDynamicWidth(12),
    overflow: "scroll",
  },

  innerContainer: {
    margin: calculateDynamicWidth(9),
    flex: 1,
    justifyContent: "space-between",
  },

  bookmark: {
    position: "absolute",
    top: 0,
    marginLeft: calculateDynamicWidth(16),
  },

  bookmarkSize: {
    width: calculateDynamicWidth(17),
    height: calculateDynamicWidth(23),
  },

  calenderContainer: {
    alignItems: "flex-end",
  },

  calendar: {
    color: Colors.hover,
    fontFamily: "Pretendard-Regular",
  },

  content: {
    color: Colors.text,
    fontSize: 18,
    fontFamily: "Pretendard-Regular",
    marginVertical: 5,
    marginHorizontal: 10,
  },

  photo: {
    width: calculateDynamicWidth(275),
    height: calculateDynamicWidth(60),
    borderRadius: calculateDynamicWidth(10),
    marginLeft: calculateDynamicWidth(5),
  },

  bottomContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
  },

  nickname: {
    color: Colors.blue500,
    fontFamily: "Pretendard-Medium",
  },

  open: {
    color: "rgba(76, 106, 255, 0.6)",
    fontFamily: "Pretendard-Medium",
  },
});
