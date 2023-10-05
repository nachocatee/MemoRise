package com.tjjhtjh.memorise.domain.memo.service.dto.response;

import com.tjjhtjh.memorise.domain.memo.repository.entity.AccessType;
import com.tjjhtjh.memorise.domain.tag.service.dto.response.TaggedTeamResponse;
import com.tjjhtjh.memorise.domain.tag.service.dto.response.TaggedUserResponse;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.nio.file.LinkOption;
import java.time.LocalDateTime;
import java.util.List;

@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class MemoDetailResponse {

    private Long userSeq;
    private String content;
    private String nickname;
    private LocalDateTime updatedAt;
    private String itemName;
    private AccessType accessType;
    private String file;
    private String itemImage;
    private Boolean isBookmarked;
    private List<TaggedUserResponse> taggedUserList;
    private List<TaggedTeamResponse> taggedTeamList;

    public MemoDetailResponse detailResponse(
            MemoDetailResponse reponse, Boolean bookmarkCheck, List<TaggedUserResponse> userList, List<TaggedTeamResponse> teamList) {
        return MemoDetailResponse.builder()
                .userSeq(reponse.getUserSeq())
                .content(reponse.getContent())
                .nickname(reponse.getNickname())
                .itemName(reponse.getItemName())
                .updatedAt(reponse.getUpdatedAt())
                .accessType(reponse.getAccessType())
                .file(reponse.getFile())
                .itemImage(reponse.getItemImage())
                .isBookmarked(bookmarkCheck)
                .taggedUserList(userList)
                .taggedTeamList(teamList)
                .build();
    }

}
