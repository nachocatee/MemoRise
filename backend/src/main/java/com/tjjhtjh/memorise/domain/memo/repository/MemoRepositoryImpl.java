package com.tjjhtjh.memorise.domain.memo.repository;

import com.querydsl.core.BooleanBuilder;
import com.querydsl.core.types.ExpressionUtils;
import com.querydsl.core.types.Projections;
import com.querydsl.core.types.dsl.CaseBuilder;
import com.querydsl.jpa.JPAExpressions;
import com.querydsl.jpa.impl.JPAQueryFactory;
import com.tjjhtjh.memorise.domain.memo.repository.entity.AccessType;
import com.tjjhtjh.memorise.domain.memo.repository.entity.Memo;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MemoDetailResponse;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MemoResponse;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MyMemoResponse;
import org.springframework.data.jpa.repository.support.QuerydslRepositorySupport;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

import static com.tjjhtjh.memorise.domain.memo.repository.entity.QBookmark.bookmark;
import static com.tjjhtjh.memorise.domain.memo.repository.entity.QMemo.memo;
import static com.tjjhtjh.memorise.domain.tag.repository.entity.QTaggedTeam.taggedTeam;
import static com.tjjhtjh.memorise.domain.tag.repository.entity.QTaggedUser.taggedUser;
import static com.tjjhtjh.memorise.domain.team.repository.entity.QTeamUser.teamUser;

@Repository
public class MemoRepositoryImpl extends QuerydslRepositorySupport implements MemoRepositoryCustom {

    private final JPAQueryFactory queryFactory;

    public MemoRepositoryImpl(JPAQueryFactory jpaQueryFactory) {
        super(Memo.class);
        this.queryFactory = jpaQueryFactory;
    }

    @Override
    public List<MemoResponse> findWrittenByMeOrOpenMemoOrTaggedMemo(Long itemSeq, Long userSeq) {
        BooleanBuilder builder = new BooleanBuilder();
        builder.and(memo.user.userSeq.eq(userSeq))  // 내가 작성했거나
                .or(memo.accessType.eq(AccessType.OPEN))  // 공개된 메모거나
                .or(memo.accessType.eq(AccessType.RESTRICT)  // 일부 공개에 내가 태그됐거나
                        .and(taggedUser.user.userSeq.eq(userSeq).and(taggedUser.memo.memoSeq.eq(memo.memoSeq))))
                .or(memo.accessType.eq(AccessType.RESTRICT) // 일부 공개에 내 그룹이 태그됐거나
                        .and(teamUser.team.teamSeq.in(JPAExpressions.select(taggedTeam.team.teamSeq).from(taggedTeam)
                                .where(taggedTeam.memo.memoSeq.eq(memo.memoSeq))).and(teamUser.user.userSeq.eq(userSeq))));

        return queryFactory.selectDistinct(Projections.fields(MemoResponse.class,
                        memo.memoSeq, memo.user.nickname.as("nickname"), memo.updatedAt, memo.content, memo.accessType, memo.file,
                        memo.item.itemImage.as("itemImage"), ExpressionUtils.as(JPAExpressions.selectOne()
                                .from(bookmark).where(bookmark.memo.memoSeq.eq(memo.memoSeq)
                                        .and(bookmark.user.userSeq.eq(userSeq))).exists(), "isBookmarked")))
                .from(memo)
                .leftJoin(taggedUser).on(memo.memoSeq.eq(taggedUser.memo.memoSeq))
                .leftJoin(teamUser).on(teamUser.team.teamSeq
                        .in(JPAExpressions.select(taggedTeam.team.teamSeq).from(taggedTeam)
                                .where(taggedTeam.memo.memoSeq.eq(memo.memoSeq))).and(teamUser.user.userSeq.eq(userSeq)))
                .where(memo.isDeleted.eq(0).and(builder).and(memo.item.itemSeq.eq(itemSeq)))
                .orderBy(memo.updatedAt.desc())
                .fetch();
    }

    @Override
    public Optional<MemoDetailResponse> detailMemo(Long memoId, Long userSeq) {
        return queryFactory.select(Projections.fields(MemoDetailResponse.class,
                        memo.user.nickname.as("nickname"), memo.updatedAt, memo.content, memo.file, memo.accessType
                        ,memo.item.itemName ,memo.item.itemImage, memo.user.userSeq))
                .from(memo)
                .leftJoin(memo.user)
                .leftJoin(memo.item)
                .where(memo.memoSeq.eq(memoId))
                .stream().findAny();
    }

    @Override
    public List<MyMemoResponse> findByMyMemoIsDeletedFalse(Long userSeq) {
        return queryFactory.select(Projections.fields
                        (MyMemoResponse.class, memo.user.nickname.as("nickname"), memo.updatedAt, memo.content, memo.accessType, memo.file,
                                memo.item.itemImage, memo.memoSeq, ExpressionUtils.as(
                                        new CaseBuilder().when(JPAExpressions.selectOne().from(bookmark)
                                        .where(bookmark.memo.memoSeq.eq(memo.memoSeq).and(bookmark.user.userSeq.eq(userSeq)))
                                        .exists()).then(true).otherwise(false), "isBookmarked")))
                .from(memo)
                .leftJoin(memo.user)
                .leftJoin(memo.item)
                .where(memo.isDeleted.eq(0).and(memo.user.userSeq.eq(userSeq)))
                .leftJoin(bookmark).on(memo.memoSeq.eq(bookmark.memo.memoSeq).and(bookmark.user.userSeq.eq(userSeq)))
                .groupBy(memo.memoSeq)
                .orderBy(memo.updatedAt.desc())
                .fetch();
    }

    @Override
    public List<MyMemoResponse> findByAllMyMemoIsDeletedFalse(Long userSeq) {
        BooleanBuilder builder = new BooleanBuilder();
        builder.and(memo.user.userSeq.eq(userSeq))  // 내가 작성했거나
                .or(memo.accessType.eq(AccessType.OPEN))  // 공개된 메모거나
                .or(memo.accessType.eq(AccessType.RESTRICT)  // 일부 공개에 내가 태그됐거나
                        .and(taggedUser.user.userSeq.eq(userSeq).and(taggedUser.memo.memoSeq.eq(memo.memoSeq))))
                .or(memo.accessType.eq(AccessType.RESTRICT) // 일부 공개에 내 그룹이 태그됐거나
                        .and(teamUser.team.teamSeq.in(JPAExpressions.select(taggedTeam.team.teamSeq).from(taggedTeam)
                                        .where(taggedTeam.memo.memoSeq.eq(memo.memoSeq))).and(teamUser.user.userSeq.eq(userSeq))));

        return queryFactory.selectDistinct(Projections.fields(MyMemoResponse.class,
                        memo.memoSeq, memo.user.nickname.as("nickname"), memo.updatedAt, memo.content, memo.accessType, memo.file,
                        memo.item.itemImage.as("itemImage"), ExpressionUtils.as(JPAExpressions.selectOne()
                                        .from(bookmark).where(bookmark.memo.memoSeq.eq(memo.memoSeq)
                                        .and(bookmark.user.userSeq.eq(userSeq))).exists(), "isBookmarked")))
                .from(memo)
                .leftJoin(taggedUser).on(memo.memoSeq.eq(taggedUser.memo.memoSeq))
                .leftJoin(teamUser).on(teamUser.team.teamSeq
                        .in(JPAExpressions.select(taggedTeam.team.teamSeq).from(taggedTeam)
                                .where(taggedTeam.memo.memoSeq.eq(memo.memoSeq))).and(teamUser.user.userSeq.eq(userSeq)))
                .where(memo.isDeleted.eq(0).and(builder))
                .orderBy(memo.updatedAt.desc())
                .fetch();
    }

    @Override
    public Long countMemoOfItem(Long itemSeq, Long userSeq) {
        BooleanBuilder builder = new BooleanBuilder();
        builder.and(memo.user.userSeq.eq(userSeq).and(memo.item.itemSeq.eq(itemSeq)))  // 내가 작성했거나
                .or(memo.accessType.eq(AccessType.OPEN).and(memo.item.itemSeq.eq(itemSeq)))  // 공개된 메모거나
                .or(memo.accessType.eq(AccessType.RESTRICT)
                        .and(memo.item.itemSeq.eq(itemSeq).and(taggedUser.user.userSeq.eq(userSeq)).and(taggedUser.memo.memoSeq.eq(memo.memoSeq))));

        return queryFactory.select(memo.memoSeq)
                .from(memo)
                .leftJoin(memo.user)
                .leftJoin(taggedUser).on(memo.memoSeq.eq(taggedUser.memo.memoSeq))
                .where(memo.isDeleted.eq(0).and(builder))
                .groupBy(memo.memoSeq)
                .stream().count();
    }

    @Override
    public Optional<Memo> findByLastSaveData(Long userSeq) {
        return queryFactory.selectFrom(memo).where(memo.user.userSeq.eq(userSeq))
                .orderBy(memo.memoSeq.desc()).limit(1).stream().findAny();
    }

}
